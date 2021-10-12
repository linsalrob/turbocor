
use std::fs::File;
use std::io::{BufReader, BufWriter, Seek, SeekFrom, Write};
use std::mem::transmute;
use std::ops::{Add, Mul};
use std::sync::{Arc, Mutex};
use std::mem::size_of;

use std::arch::x86_64::{
    __m256,
    __m256i,
    _mm256_add_ps,
    _mm256_castps256_ps128,
    _mm256_extractf128_ps,
    _mm256_fmadd_ps,
    _mm256_mul_ps,
    _mm256_set1_ps,
    _mm256_set_ps,
    _mm256_testc_si256,
    _mm_add_ps,
    _mm_cvtss_f32,
    _mm_movehl_ps,
    _mm_shuffle_ps,
};

extern crate byteorder;
use byteorder::{BigEndian, WriteBytesExt, ReadBytesExt};

extern crate indicatif;
// use indicatif::{ProgressIterator, ParallelProgressIterator};
use indicatif::ProgressIterator;

extern crate thread_local;
use thread_local::ThreadLocal;

extern crate num_traits;
use num_traits::identities::Zero;

extern crate clap;
use clap::{Arg, App, SubCommand};

extern crate tempfile;
use tempfile::tempfile;

extern crate rayon;
use rayon::prelude::*;

// Wrapper for __m256
#[derive(Clone, Copy, Debug)]
struct F32x8 {
    x: __m256
}

impl F32x8 {
    fn from_slice(xs: &[f32; 8]) -> Self {
        // this segfault, not sure why
        // return F32x8{x: unsafe { *transmute::<&[f32; 8], &__m256>(xs) } };
        return F32x8{x: unsafe { _mm256_set_ps(xs[0], xs[1], xs[2], xs[3], xs[4], xs[5], xs[6], xs[7]) } };
    }

    // Fused multiply-add computing self*v + b
    #[inline(always)]
    fn fma(&self, v: &Self, b: &Self) -> F32x8 {
        return F32x8{x: unsafe { _mm256_fmadd_ps(self.x, v.x, b.x) }};
    }

    // horizontal sum
    // from here: https://stackoverflow.com/a/13222410
    fn sum(&self) -> f32 {
        unsafe {
            let hiquad = _mm256_extractf128_ps(self.x, 1);
            let loquad = _mm256_castps256_ps128(self.x);
            let sumquad = _mm_add_ps(hiquad, loquad);

            let lodual = sumquad;
            let hidual = _mm_movehl_ps(sumquad, sumquad);
            let sumdual = _mm_add_ps(lodual, hidual);

            let lo = sumdual;
            let hi = _mm_shuffle_ps(sumdual, sumdual, 1);
            let sum = _mm_add_ps(lo, hi);

            return _mm_cvtss_f32(sum);
        }
    }
}

impl Add<F32x8> for F32x8 {
    type Output = Self;

    fn add(self, y: F32x8) -> F32x8 {
        return F32x8{x: unsafe { _mm256_add_ps(self.x, y.x) } };
    }
}

impl Mul<F32x8> for F32x8 {
    type Output = Self;

    fn mul(self, y: F32x8) -> F32x8 {
        return F32x8{x: unsafe { _mm256_mul_ps(self.x, y.x) } };
    }
}

impl Zero for F32x8 {
    fn zero() -> F32x8 {
        return F32x8{x: unsafe { _mm256_set1_ps(0.0) } };
    }

    fn is_zero(&self) -> bool {
        let self_256i = unsafe { transmute::<&__m256, &__m256i>(&self.x) };
        return unsafe { _mm256_testc_si256(*self_256i, *self_256i) } == 1;
    }
}


// Read HDF5 feature matrix into a padded AVX (F32x8) matrix.
fn read_feature_matrix(filename: &str) -> hdf5::Result<(Vec<F32x8>, usize, usize, usize)> {
    let file = hdf5::File::open(filename)?;
    let x_ds = file.dataset("X")?;

    let mut x = x_ds.read_2d::<f32>()?;

    let m = x.shape()[0];
    let n = x.shape()[1];

    // center by subtracting row-means
    x.outer_iter_mut().for_each(|mut row| {
        let mu = row.fold(0.0, |a, b| { a + b }) / (n as f32);
        row.iter_mut().for_each(|v| *v -= mu );
    });

    // Pad and convert to a f32x8 matrix
    let n_avx = ((n-1) / 8)+1;

    let mut x_avx: Vec<F32x8> = vec![F32x8::zero(); m*n_avx];

    // let mut x_avx = Array2::<F32x8>::zeros((m, n_avx));
    let mut buf: [f32; 8] = [0.0; 8];

    for (i, row) in x.outer_iter().enumerate() {
        let mut k = 0;
        for (j, value) in row.iter().enumerate() {
            k = j % 8;
            buf[k] = *value;
            if k == 7 {
                x_avx[i*n_avx + j/8] = F32x8::from_slice(&buf);
            }
        }

        // handle remainder
        if k != 7 {
            for l in k+1..8 {
                buf[l] = 0.0;
            }
            x_avx[i*n_avx + (n-1)/8] = F32x8::from_slice(&buf);
        }
    }

    return Ok((x_avx, m, n, n_avx));
}


// standard deviation of centered (zero-mean) rows
fn l2norms(x: &Vec<F32x8>, n_avx: usize) -> Vec<f32> {
    let m = x.len() / n_avx;
    let mut norms: Vec<f32> = Vec::with_capacity(m);
    for row in x.chunks_exact(n_avx) {
        norms.push(sum_squares(row).sqrt());
    }
    return norms;
}


fn dot(u: &[F32x8], v: &[F32x8]) -> f32 {
    let mut accum = F32x8::zero();
    u.iter().zip(v).for_each(|(u_i, v_i)| { accum = u_i.fma(v_i, &accum) });
    return accum.sum();
}


fn sum_squares(u: &[F32x8]) -> f32 {
    return u.iter().fold(F32x8::zero(), |a, b| { a + (*b * *b) }).sum();
}


// Compute a sparse correlation matrix
fn correlation_matrix(
        x: &Vec<F32x8>, norms: &Vec<f32>, n_avx: usize,
        tmpfile: &mut File, lower_bound: f32) -> (Vec<i32>, Vec<i32>, Vec<f32>) {

    let m = x.len() / n_avx;

    // Each thread adds results to its own buffer. When a buffer is full, it locks
    // the temporary file and purges its buffer.
    const BUFSIZE: usize = 1_000_000;
    let buffers: Arc<ThreadLocal<Mutex<Vec<(i32, i32, f32)>>>> = Arc::new(ThreadLocal::new());
    let tmpfile = Arc::new(Mutex::new(tmpfile));

    // TODO: fancy progress bar

    x.par_chunks_exact(n_avx).zip(norms).enumerate().for_each(|(i, (u, u_norm))| {
        let buffer = buffers.clone();

        ((i+1)..m).zip(x[(i+1)*n_avx..].chunks_exact(n_avx).zip(norms[(i+1)..].iter())).for_each(|(j, (v, v_norm))| {
            let c = dot(u, v) / (u_norm * v_norm);

            if c.abs() < lower_bound {
                return
            }

            let mut buffer = buffer.get_or(|| Mutex::new(Vec::with_capacity(BUFSIZE))).lock().unwrap();

            buffer.push((i as i32, j as i32, c));
            if buffer.len() >= BUFSIZE {
                let mut tmpfile = tmpfile.lock().unwrap();
                write_temporary_entries(*tmpfile, &buffer);
                buffer.clear();
            }
        });
    });

    // handle partially filled buffers
    println!("flushing buffers...");
    let mut tmpfile = tmpfile.lock().unwrap();
    let buffers = Arc::try_unwrap(buffers).unwrap();
    let mut buffer_count = 0;
    buffers.into_iter().for_each(|buffer| {
        buffer_count += 1;
        let mut buffer = buffer.lock().unwrap();
        write_temporary_entries(*tmpfile, &buffer);
        buffer.clear();
    });
    println!("done.");

    tmpfile.flush().unwrap();
    tmpfile.seek(SeekFrom::Start(0)).unwrap();

    // inspect file size to count the total entries
    let filebytes = tmpfile.metadata().unwrap().len() as usize;
    let bytes_per_entry = 2*size_of::<i32>() + size_of::<f32>();
    assert!(filebytes % bytes_per_entry == 0);
    let nnz = filebytes / bytes_per_entry;

    return read_temporary_entries(*tmpfile, nnz);
}


fn read_temporary_entries(input: &mut File, nnz: usize) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let mut input = BufReader::new(input);

    let mut i_idx = Vec::<i32>::with_capacity(nnz);
    let mut j_idx = Vec::<i32>::with_capacity(nnz);
    let mut v_idx = Vec::<f32>::with_capacity(nnz);

    println!("reading back {} matrix entries...", nnz);
    for _ in 0..nnz {
        i_idx.push(input.read_i32::<BigEndian>().unwrap());
        j_idx.push(input.read_i32::<BigEndian>().unwrap());
        v_idx.push(input.read_f32::<BigEndian>().unwrap());
    }
    println!("done.");

    return (i_idx, j_idx, v_idx);
}


fn write_temporary_entries(output: &mut File, buffer: &Vec<(i32, i32, f32)>) {
    let mut output = BufWriter::new(output);

    for (i, j, v) in buffer.iter() {
        output.write_i32::<BigEndian>(*i).unwrap();
        output.write_i32::<BigEndian>(*j).unwrap();
        output.write_f32::<BigEndian>(*v).unwrap();
    }
}


// Output correlation matrix in COO format to an HDF5 file
fn write_correlation_matrix(filename: &str, i_idx: &Vec<i32>, j_idx: &Vec<i32>, v_idx: &Vec<f32>) -> hdf5::Result<()>  {
    assert!(i_idx.len() == j_idx.len());
    assert!(j_idx.len() == v_idx.len());
    let nnz = i_idx.len();
    let output = hdf5::File::create(filename)?;
    let i_ds = output.new_dataset::<i32>().create("I", nnz)?;
    i_ds.write(i_idx)?;
    let j_ds = output.new_dataset::<i32>().create("J", nnz)?;
    j_ds.write(j_idx)?;
    let v_ds = output.new_dataset::<f32>().create("V", nnz)?;
    v_ds.write(v_idx)?;

    return Ok(());
}


fn main() {
    // TODO: What are we actually calling this thing?
    let matches = App::new("")
        .about("Brute-force computation of correlation matrices.")
        .subcommand(SubCommand::with_name("compute")
            .arg(Arg::with_name("input")
                .help("Feature matrix in HDF5 format.")
                .value_name("input-features.h5")
                .required(true)
                .index(1))
            .arg(Arg::with_name("output")
                .help("Output sparse COO correlation matrix in HDF5 format.")
                .value_name("output-correlation.h5")
                .required(true)
                .index(2))
            .arg(Arg::with_name("lower-bound")
                .help("Suppress abs correlations below this number.")
                .takes_value(true)
                .value_name("C"))
            .arg(Arg::with_name("temp")
                .help("Path to temporary file to use.")
                .takes_value(true)
                .value_name("TEMPFILE"))
            .arg(Arg::with_name("threads")
                .help("Number of threads to use.")
                .takes_value(true)
                .value_name("THREADS")))
        .subcommand(SubCommand::with_name("topk"))
        .get_matches();


    if let Some(matches) = matches.subcommand_matches("compute") {
        let input_filename = matches.value_of("input").unwrap();
        let output_filename = matches.value_of("output").unwrap();
        let lower_bound = matches.value_of("lower-bound").unwrap_or("0.8").parse::<f32>().unwrap();

        println!("lower bound: {}", lower_bound);

        // TODO: handle --threads option

        let mut tmpfile = if let Some(temp_filename) = matches.value_of("temp") {
            File::create(temp_filename).unwrap()
        } else {
            tempfile().unwrap()
        };

        println!("here");
        let (x, m, n, n_avx) = read_feature_matrix(input_filename).unwrap();
        let norms = l2norms(&x, n_avx);
        println!("Input matrix: {} features, {} observations", m, n);

        let (i_idx, j_idx, v_idx) = correlation_matrix(&x, &norms, n_avx, &mut tmpfile, lower_bound);
        write_correlation_matrix(output_filename, &i_idx, &j_idx, &v_idx).unwrap();

    } else if let Some(matches) = matches.subcommand_matches("topk") {
        // TODO:
        //  - Read hdf5 sparse correlation matrix.
        //  - To partial proxy sort.
        //  - Report results to csv
    } else {
        // TODO: error
    }
}