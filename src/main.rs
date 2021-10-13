
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
use indicatif::{ProgressBar, ProgressStyle, ParallelProgressIterator};

extern crate thread_local;
use thread_local::ThreadLocal;

extern crate num_traits;
use num_traits::identities::Zero;

extern crate clap;
use clap::{Arg, App, AppSettings, SubCommand};

extern crate tempfile;
use tempfile::tempfile;

extern crate rayon;
use rayon::prelude::*;

extern crate partial_sort;
use partial_sort::PartialSort;

// Wrapper for __m256
#[derive(Clone, Copy, Debug)]
struct F32x8 {
    x: __m256
}

impl F32x8 {
    fn from_slice(xs: &[f32; 8]) -> Self {
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
fn read_feature_matrix(
        filename: &str, dataset_name: &str, transpose: bool) -> hdf5::Result<(Vec<F32x8>, usize, usize, usize)> {
    let file = hdf5::File::open(filename)?;
    let x_ds = file.dataset(dataset_name)?;

    let mut x = x_ds.read_2d::<f32>()?;
    if transpose {
        x = x.reversed_axes();
    }

    let m = x.shape()[0];
    let n = x.shape()[1];

    // center by subtracting row-means, then divide by row l2 norms
    x.outer_iter_mut().for_each(|mut row| {
        let mu = row.fold(0.0, |a, b| { a + b }) / (n as f32);
        row.iter_mut().for_each(|v| *v -= mu );
        let l2 = row.fold(0.0, |accum, v| {accum + (v * v)} ).sqrt();
        row.iter_mut().for_each(|v| *v /= l2 );
    });

    // Pad and convert to a f32x8 matrix
    let n_avx = ((n-1) / 8)+1;

    let mut x_avx: Vec<F32x8> = vec![F32x8::zero(); m*n_avx];
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


fn dot(u: &[F32x8], v: &[F32x8]) -> f32 {
    return u.iter().zip(v).fold(F32x8::zero(), |accum, (u_i, v_i)| { u_i.fma(v_i, &accum) }).sum();
}


// Compute a sparse correlation matrix
fn correlation_matrix(
        x: &Vec<F32x8>, n_avx: usize,
        tmpfile: &mut File, lower_bound: f32) -> (Vec<i32>, Vec<i32>, Vec<f32>) {

    let m = x.len() / n_avx;

    // Each thread adds results to its own buffer. When a buffer is full, it locks
    // the temporary file and purges its buffer.
    const BUFSIZE: usize = 1_000_000;
    let buffers: Arc<ThreadLocal<Mutex<Vec<(i32, i32, f32)>>>> = Arc::new(ThreadLocal::new());
    let tmpfile = Arc::new(Mutex::new(tmpfile));

    let prog = ProgressBar::new(m as u64);
    prog.set_style(ProgressStyle::default_bar()
        .template("Computing correlation matrix: {bar:80} {eta_precise}"));

    x.par_chunks_exact(n_avx).enumerate().progress_with(prog).for_each(|(i, u)| {
        let buffer = buffers.clone();

        ((i+1)..m).zip(x[(i+1)*n_avx..].chunks_exact(n_avx)).for_each(|(j, v)| {
            let c = dot(u, v);

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
    buffers.into_iter().for_each(|buffer| {
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
    let mut vs = Vec::<f32>::with_capacity(nnz);

    println!("reading back {} matrix entries...", nnz);
    for _ in 0..nnz {
        i_idx.push(input.read_i32::<BigEndian>().unwrap());
        j_idx.push(input.read_i32::<BigEndian>().unwrap());
        vs.push(input.read_f32::<BigEndian>().unwrap());
    }
    println!("done.");

    return (i_idx, j_idx, vs);
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
fn write_correlation_matrix(filename: &str, i_idx: &Vec<i32>, j_idx: &Vec<i32>, vs: &Vec<f32>) -> hdf5::Result<()>  {
    assert!(i_idx.len() == j_idx.len());
    assert!(j_idx.len() == vs.len());
    let nnz = i_idx.len();
    let output = hdf5::File::create(filename)?;
    let i_ds = output.new_dataset::<i32>().create("I", nnz)?;
    i_ds.write(i_idx)?;
    let j_ds = output.new_dataset::<i32>().create("J", nnz)?;
    j_ds.write(j_idx)?;
    let v_ds = output.new_dataset::<f32>().create("V", nnz)?;
    v_ds.write(vs)?;

    return Ok(());
}


fn read_correlation_matrix(filename: &str) -> hdf5::Result<(Vec<i32>, Vec<i32>, Vec<f32>)> {

    let file = hdf5::File::open(filename)?;

    let i_ds = file.dataset("I")?;
    let i_idx = i_ds.read_raw()?;

    let j_ds = file.dataset("J")?;
    let j_idx = j_ds.read_raw()?;

    let v_ds = file.dataset("V")?;
    let vs = v_ds.read_raw()?;

    return Ok((i_idx, j_idx, vs))
}


fn main() {
    let matches = App::new("turbocor")
        .about("Brute-force computation of correlation matrices.")
        .setting(AppSettings::ArgRequiredElseHelp)
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
                .long("lower-bound")
                .takes_value(true)
                .value_name("C"))
            .arg(Arg::with_name("dataset")
                .help("Name of dataset in input HDF5 file.")
                .long("dataset")
                .takes_value(true)
                .value_name("DATASET"))
            .arg(Arg::with_name("transpose")
                .help("Dataset matrix should be transposed")
                .long("transpose"))
            .arg(Arg::with_name("temp")
                .help("Path to temporary file to use.")
                .takes_value(true)
                .value_name("TEMPFILE"))
            .arg(Arg::with_name("threads")
                .help("Number of threads to use.")
                .takes_value(true)
                .value_name("THREADS")))
        .subcommand(SubCommand::with_name("topk")
            .help("Print the top-k correlations (or anti-correlations)")
            .arg(Arg::with_name("k")
                .help("Maximum number of correlated pairs to print.")
                .required(true)
                .value_name("K")
                .index(1))
            .arg(Arg::with_name("input")
                .help("Sparse COO correlation matrix in HDF5 format (generated by the `compute` command).")
                .value_name("correlations.h5")
                .required(true)
                .index(2)))
        .get_matches();

    if let Some(matches) = matches.subcommand_matches("compute") {
        let input_filename = matches.value_of("input").unwrap();
        let output_filename = matches.value_of("output").unwrap();
        let lower_bound = matches.value_of("lower-bound").unwrap_or("0.8").parse::<f32>().unwrap();
        let dataset_name = matches.value_of("dataset").unwrap_or("X");
        let transpose = matches.is_present("transpose");

        println!("lower bound: {}", lower_bound);

        // TODO: handle --threads option

        let mut tmpfile = if let Some(temp_filename) = matches.value_of("temp") {
            File::create(temp_filename).unwrap()
        } else {
            tempfile().unwrap()
        };

        let (x, m, n, n_avx) = read_feature_matrix(input_filename, dataset_name, transpose).unwrap();
        println!("Input matrix: {} features, {} observations", m, n);

        let (i_idx, j_idx, vs) = correlation_matrix(&x, n_avx, &mut tmpfile, lower_bound);
        write_correlation_matrix(output_filename, &i_idx, &j_idx, &vs).unwrap();

    } else if let Some(matches) = matches.subcommand_matches("topk") {
        let input_filename = matches.value_of("input").unwrap();
        let k = matches.value_of("k").unwrap().parse::<usize>().unwrap();

        let (i_idx, j_idx, vs) = read_correlation_matrix(input_filename).unwrap();

        let nnz = vs.len();

        let mut perm: Vec<usize> = (0..nnz).collect();
        perm.partial_sort(k.min(nnz), |i, j| vs[*j].abs().partial_cmp(&vs[*i].abs()).unwrap() );

        for p in perm[0..k.min(nnz)].iter() {
            println!("{},{},{}", i_idx[*p], j_idx[*p], vs[*p]);
        }
    } else {
        panic!("Subcommand must be used.")
    }
}