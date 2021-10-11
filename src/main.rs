
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::mem::transmute;
use std::ops::Add;
use std::sync::{Arc, Mutex};
use std::mem::size_of;

use std::arch::x86_64::{
    __m256,
    __m256i,
    _mm256_add_ps,
    _mm256_castps256_ps128,
    _mm256_extractf128_ps,
    _mm256_fmadd_ps,
    _mm256_set1_ps,
    _mm256_testc_si256,
    _mm_add_ps,
    _mm_cvtss_f32,
    _mm_movehl_ps,
    _mm_shuffle_ps,
};

extern crate byteorder;
use byteorder::{BigEndian, WriteBytesExt, ReadBytesExt};

extern crate thread_local;
use thread_local::ThreadLocal;

extern crate num_traits;
use num_traits::identities::Zero;

// Wrapper for __m256
#[derive(Clone, Copy, Debug)]
struct F32x8 {
    x: __m256
}

impl F32x8 {
    fn from_slice(xs: &[f32; 8]) -> Self {
        return F32x8{x: unsafe { *transmute::<&[f32; 8], &__m256>(xs) } };
    }


    // Fused multiply-add computing self*v + b
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

impl Zero for F32x8 {
    fn zero() -> F32x8 {
        return F32x8{x: unsafe { _mm256_set1_ps(0.0) } };
    }

    fn is_zero(&self) -> bool {
        let self_256i = unsafe { transmute::<&__m256, &__m256i>(&self.x) };
        return unsafe { _mm256_testc_si256(*self_256i, *self_256i) } == 1;
    }
}

extern crate clap;
use clap::{Arg, App, SubCommand};

extern crate tempfile;
use tempfile::tempfile;

extern crate ndarray;
// use ndarray::prelude::*;
use ndarray::{Array2, ArrayView, Axis, Ix1, Zip};
use ndarray::parallel::prelude::*;


// Read HDF5 feature matrix into a padded AVX (F32x8) matrix.
fn read_feature_matrix(filename: &str) -> hdf5::Result<Array2<F32x8>> {
    let file = hdf5::File::open(filename)?;
    let Xds = file.dataset("X")?;

    let X = Xds.read_2d::<f32>()?;

    // TODO: we have to do some normalization stuff for this to work. What was it?

    let m = X.shape()[0];
    let n = X.shape()[1];

    // Pad and convert to a f32x8 matrix
    let n_avx = ((n-1) / 8)+1;
    let mut X_avx = Array2::<F32x8>::zeros((m, n_avx));
    let mut buf: [f32; 8] = [0.0; 8];

    for (i, row) in X.outer_iter().enumerate() {
        let mut k = 0;
        for (j, value) in row.iter().enumerate() {
            k = j % 8;
            buf[k] = *value;
            if k == 7 {
                X_avx[[i, j/8]] = F32x8::from_slice(&buf);
            }
        }

        // handle remainder
        if k != 7 {
            for l in k+1..8 {
                buf[l] = 0.0;
            }
            X_avx[[i, (n-1)/8]] = F32x8::from_slice(&buf);
        }
    }

    return Ok(X_avx);
}


// dot product function using fmadd and and fancy horizontal sum
fn dot<'a>(u: ArrayView<'a, F32x8, Ix1>, v: ArrayView<'a, F32x8, Ix1>) -> f32 {
    let mut accum = F32x8::zero();
    Zip::from(u).and(v).for_each(|x, y| { accum = x.fma(y, &accum) });
    return accum.sum();
}


// Compute a sparse correlation matrix
fn correlation_matrix(X: &Array2<F32x8>, tmpfile: &mut File, lower_bound: f32) -> (Vec<i32>, Vec<i32>, Vec<f32>) {

    // Each thread adds results to its own buffer. When a buffer is full, it locks
    // the temporary file and purges its buffer.
    const BUFSIZE: usize = 10000;
    let buffers: Arc<ThreadLocal<Mutex<Vec<(i32, i32, f32)>>>> = Arc::new(ThreadLocal::new());
    let tmpfile = Arc::new(Mutex::new(tmpfile));

    // TODO: fancy progress bar

    X.axis_iter(Axis(0)).into_par_iter().enumerate().for_each(|(i, u)| {
        let buffer = buffers.clone();

        X.axis_iter(Axis(0)).enumerate().for_each(|(j, v)| {
            // only compute the upper triangular matrix
            if j <= i {
                return;
            }

            let c = dot(u, v);
            // TODO: normalization stuff?

            if c < lower_bound {
                return
            }

            let mut buffer = buffer.get_or(|| Mutex::new(Vec::with_capacity(BUFSIZE))).lock().unwrap();

            buffer.push((i as i32, j as i32, c));
            if buffer.len() >= BUFSIZE {
                let mut tmpfile = tmpfile.lock().unwrap();
                for (i, j, v) in buffer.iter() {
                    tmpfile.write_i32::<BigEndian>(*i).unwrap();
                    tmpfile.write_i32::<BigEndian>(*j).unwrap();
                    tmpfile.write_f32::<BigEndian>(*v).unwrap();
                }
                buffer.clear();
            }
        })
    });

    // handle partially filled buffers
    let mut tmpfile = tmpfile.lock().unwrap();
    let buffers = Arc::try_unwrap(buffers).unwrap();
    let mut buffer_count = 0;
    buffers.into_iter().for_each(|buffer| {
        buffer_count += 1;
        let mut buffer = buffer.lock().unwrap();
            for (i, j, v) in buffer.iter() {
                tmpfile.write_i32::<BigEndian>(*i).unwrap();
                tmpfile.write_i32::<BigEndian>(*j).unwrap();
                tmpfile.write_f32::<BigEndian>(*v).unwrap();
            }
            buffer.clear();
    });
    dbg!(buffer_count);

    tmpfile.flush().unwrap();
    tmpfile.seek(SeekFrom::Start(0)).unwrap();

    // inspect file size to count the total entries
    let filebytes = tmpfile.metadata().unwrap().len() as usize;
    let bytes_per_entry = 2*size_of::<i32>() + size_of::<f32>();
    assert!(filebytes % bytes_per_entry == 0);
    let nnz = filebytes / bytes_per_entry;

    let mut Is = Vec::<i32>::with_capacity(nnz);
    let mut Js = Vec::<i32>::with_capacity(nnz);
    let mut Vs = Vec::<f32>::with_capacity(nnz);

    for _ in 0..nnz {
        Is.push(tmpfile.read_i32::<BigEndian>().unwrap());
        Js.push(tmpfile.read_i32::<BigEndian>().unwrap());
        Vs.push(tmpfile.read_f32::<BigEndian>().unwrap());
    }

    return (Is, Js, Vs);
}


// Output correlation matrix in COO format to an HDF5 file
fn write_correlation_matrix(filename: &str, Is: &Vec<i32>, Js: &Vec<i32>, Vs: &Vec<f32>) -> hdf5::Result<()>  {
    assert!(Is.len() == Js.len());
    assert!(Js.len() == Vs.len());
    let nnz = Is.len();
    let output = hdf5::File::create(filename)?;
    let i_ds = output.new_dataset::<i32>().create("I", nnz)?;
    i_ds.write(Is)?;
    let j_ds = output.new_dataset::<i32>().create("J", nnz)?;
    j_ds.write(Js)?;
    let v_ds = output.new_dataset::<f32>().create("V", nnz)?;
    v_ds.write(Vs)?;

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
        let lower_bound = matches.value_of("lower-bound").unwrap().parse::<f32>().unwrap();

        // TODO: handle --threads option

        let mut tmpfile = if let Some(temp_filename) = matches.value_of("temp") {
            File::create(temp_filename).unwrap()
        } else {
            tempfile().unwrap()
        };

        let X = read_feature_matrix(input_filename).unwrap();

        let (Is, Js, Vs) = correlation_matrix(&X, &mut tmpfile, lower_bound);
        write_correlation_matrix(output_filename, Is, Js, Vs);

    } else if let Some(matches) = matches.subcommand_matches("topk") {
        // TODO:
        //  - Read hdf5 sparse correlation matrix.
        //  - To partial proxy sort.
        //  - Report results to csv
    } else {
        // TODO: error
    }

    println!("Hello, world!");
}
