use crate::csr::*;
use crate::helper::split_offsets;
use itertools::{iproduct, Itertools};
use lazy_static::lazy_static;
use ngrams::Ngram;
use num::Num;
use polars::prelude::*;
use rayon::prelude::*;
use std::{
    collections::HashMap,
    ops::{AddAssign, MulAssign},
};

lazy_static! {
    static ref MAPPING: HashMap<Vec<char>, usize> = {
        let ngrams = generate_ngrams();
        generate_ngram_index_mapping(ngrams)
    };
}

fn generate_ngrams() -> Vec<Vec<char>> {
    let alphabet = ('a'..='z').collect::<Vec<char>>();

    iproduct!(alphabet.iter(), alphabet.iter(), alphabet.iter())
        .map(|(a, b, c)| vec![*a, *b, *c])
        .collect::<Vec<Vec<char>>>()
}

fn generate_ngram_index_mapping(ngrams: Vec<Vec<char>>) -> HashMap<Vec<char>, usize> {
    let mut ngram_index_mapping = HashMap::new();
    // index 0 is reserved for non-existing mapping
    for (index, ngram) in ngrams.into_iter().enumerate() {
        ngram_index_mapping.insert(ngram, index + 1);
    }
    ngram_index_mapping
}

fn transform<T: Num>(sa: &Series) -> CsrMatBase<T> {
    let mut indptr = vec![0];
    let mut indices = vec![];
    let mut data = vec![];

    let sa = sa.str().unwrap();

    for s in sa.into_iter() {
        let s = s.unwrap();
        let ngram = s.chars().ngrams(3).pad();
        let mut nnz = 0;
        // TODO: eventually we could use tfidf or similar
        // and add up the occurences of the ngram
        // only add ngram once
        // make sure to make match this with the csr::normlize_rows method
        for ngram_value in ngram.unique() {
            if let Some(index) = MAPPING.get(&ngram_value) {
                nnz += 1;
                indices.push(*index);
                data.push(T::one());
            }
        }
        indptr.push(indptr.last().unwrap() + nnz);
    }

    CsrMatBase::new(indptr, indices, data, sa.len(), MAPPING.len())
}

fn sparse_dot_topn<T: Num + Copy + MulAssign + AddAssign + PartialOrd>(
    a: &CsrMatBase<T>,
    b: &CsrMatBase<T>,
    ntop: usize,
) -> CsrMatBase<T> {
    let mut indptr = Vec::with_capacity(a.rows + 1);
    let mut indices = Vec::with_capacity(a.rows * ntop);
    let mut data = Vec::with_capacity(a.rows * ntop);

    indptr.push(0);

    let rows = a.rows;
    let cols = b.cols;

    let am = a.rows;
    let an = a.cols;
    let bm = b.rows;
    let bn = b.cols;

    assert_eq!(an, bm);

    for i in 0..am {
        let mut sums = vec![T::zero(); bn];
        let mut candidates = Vec::<(usize, T)>::with_capacity(ntop);

        let jj_start = a.indptr[i];
        let jj_end = a.indptr[i + 1];

        for jj in jj_start..jj_end {
            let j = a.indices[jj];
            let v = a.data[jj];

            let kk_start = b.indptr[j];
            let kk_end = b.indptr[j + 1];

            for kk in kk_start..kk_end {
                let k = b.indices[kk];
                let w = b.data[kk];

                sums[k] += v * w;
            }
        }

        for (j, score) in sums.iter().enumerate() {
            if *score != T::zero() {
                candidates.push((j, *score));
            }
        }

        let mut nnz = 0;

        let topn_candidates = if candidates.len() <= ntop {
            candidates
        } else {
            candidates
                .select_nth_unstable_by(ntop, |a, b| b.1.partial_cmp(&a.1).unwrap())
                .0
                .to_vec()
        };

        for (j, v) in topn_candidates {
            indices.push(j);
            data.push(v);
            nnz += 1;
        }

        // even if thare are no candidates (i.e. nnz=0), we push the row
        // and potentially produce a piecewise constant indptr
        indptr.push(indptr.last().unwrap() + nnz);
    }

    CsrMatBase::new(indptr, indices, data, rows, cols)
}

fn left_parallel_sparse_dot_top_n<
    T: Num + Copy + MulAssign + AddAssign + PartialOrd + Send + Sync,
>(
    a: &CsrMatBase<T>,
    b: &CsrMatBase<T>,
    ntop: usize,
    threads: usize,
) -> (Vec<(usize, usize)>, Vec<CsrMatBase<T>>) {
    assert_eq!(a.cols, b.rows);

    let offsets = split_offsets(a.rows, threads);

    let csr_batches = offsets
        .par_iter()
        .map(|(offset, len)| {
            let a_batch = a.slice(*offset, *len);
            sparse_dot_topn(&a_batch, b, ntop)
        })
        .collect::<Vec<CsrMatBase<T>>>();

    (offsets, csr_batches)
}

fn right_parallel_sparse_dot_top_n<
    T: Num + Copy + MulAssign + AddAssign + PartialOrd + Send + Sync,
>(
    a: &CsrMatBase<T>,
    b: &CsrMatBase<T>,
    ntop: usize,
    threads: usize,
) -> (Vec<(usize, usize)>, Vec<CsrMatBase<T>>) {
    // b comes on non transposed

    assert_eq!(a.cols, b.cols);

    let offsets = split_offsets(b.rows, threads);

    let csr_batches = offsets
        .par_iter()
        .map(|(offset, len)| {
            let b_batch = b.slice(*offset, *len).transpose();
            sparse_dot_topn(a, &b_batch, ntop)
        })
        .collect::<Vec<CsrMatBase<T>>>();

    (offsets, csr_batches)
}

fn chunked_csr_to_df<T>(
    offsets: Vec<(usize, usize)>,
    csr_batches: Vec<CsrMatBase<T>>,
) -> PolarsResult<DataFrame>
where
    T: Into<f32> + Copy,
{
    let mut rows: Vec<i32> = vec![];
    let mut indices: Vec<i32> = vec![];
    let mut data: Vec<f32> = vec![];

    for (offset, batch) in offsets.iter().zip(csr_batches.iter()) {
        for i in 0..batch.rows {
            let jj_start = batch.indptr[i];
            let jj_end = batch.indptr[i + 1];

            for jj in jj_start..jj_end {
                rows.push(i as i32 + offset.0 as i32);
                indices.push(batch.indices[jj] as i32);
                data.push(batch.data[jj].into());
            }
        }
    }

    DataFrame::new(vec![
        Series::new("row".into(), rows),
        Series::new("col".into(), &indices),
        Series::new("sim".into(), data),
    ])
}

fn csr_to_df<T>(csr: CsrMatBase<T>) -> PolarsResult<DataFrame>
where
    T: Into<f32> + Copy,
{
    let mut rows: Vec<i32> = vec![];
    let mut indices: Vec<i32> = vec![];
    let mut data: Vec<f32> = vec![];

    for i in 0..csr.rows {
        let jj_start = csr.indptr[i];
        let jj_end = csr.indptr[i + 1];

        for jj in jj_start..jj_end {
            rows.push(i as i32);
            indices.push(csr.indices[jj] as i32);
            data.push(csr.data[jj].into());
        }
    }

    DataFrame::new(vec![
        Series::new("row".into(), rows),
        Series::new("col".into(), &indices),
        Series::new("sim".into(), data),
    ])
}

fn compute_cossim<T>(
    a: CsrMatBase<T>,
    b: CsrMatBase<T>,
    ntop: usize,
    threads: usize,
    parallelize_left: bool,
) -> PolarsResult<DataFrame>
where
    T: Num + Copy + MulAssign + AddAssign + PartialOrd + Send + Sync + Into<f32>,
{
    if parallelize_left {
        let b = b.transpose();
        let (offsets, batches) = left_parallel_sparse_dot_top_n(&a, &b, ntop, threads);

        chunked_csr_to_df(offsets, batches)
    } else {
        let (_, batches) = right_parallel_sparse_dot_top_n(&a, &b, ntop, threads);

        // reduce the batches to the top n matches
        let reduced_csr = topn_from_csr_batches(batches, ntop);

        csr_to_df(reduced_csr)
    }
}

pub(super) fn awesome_cossim(
    df_left: DataFrame,
    df_right: DataFrame,
    col_left: &str,
    col_right: &str,
    ntop: usize,
    threads: Option<usize>,
    normalize: Option<bool>,
    parallelize_left: Option<bool>,
) -> PolarsResult<DataFrame> {
    let threads = threads.unwrap_or(rayon::current_num_threads());
    let normalize = normalize.unwrap_or(false);
    let parallelize_left = parallelize_left.unwrap_or(true);

    let sa = df_left.column(col_left).unwrap();
    let sb = df_right.column(col_right).unwrap();

    match normalize {
        true => {
            let mut a = transform::<f32>(sa);
            let mut b = transform::<f32>(sb);
            a.normalize_rows();
            b.normalize_rows();
            compute_cossim(a, b, ntop, threads, parallelize_left)
        }
        false => {
            let a = transform::<u16>(sa);
            let b = transform::<u16>(sb);
            compute_cossim(a, b, ntop, threads, parallelize_left)
        }
    }
}
