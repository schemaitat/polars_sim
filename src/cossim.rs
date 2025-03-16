use crate::csr::*;

use crate::helper::split_offsets;
use itertools::{iproduct, Itertools};
use ngrams::Ngram;
use polars::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

type ICsrMat = CsrMatBase<u16, u32, u32>;
type FCsrMat = CsrMatBase<f32, u32, u32>;

fn generate_ngrams() -> Vec<Vec<char>> {
    let alphabet = ('a'..='z').collect::<Vec<char>>();

    iproduct!(alphabet.iter(), alphabet.iter(), alphabet.iter())
        .map(|(a, b, c)| vec![*a, *b, *c])
        .collect::<Vec<Vec<char>>>()
}

fn generate_ngram_index_mapping<J>(ngrams: Vec<Vec<char>>) -> HashMap<Vec<char>, J>
where
    J: IndexStorage,
{
    let mut ngram_index_mapping = HashMap::new();
    let mut cur_index = J::zero(); // Start from 0
    for ngram in ngrams.into_iter() {
        ngram_index_mapping.insert(ngram, cur_index);
        cur_index = cur_index + J::one();
    }
    ngram_index_mapping
}

fn transform<T, I, J>(sa: &Series) -> CsrMatBase<T, I, J>
where
    T: DataStorage,
    I: IndPtrStorage,
    J: IndexStorage,
{
    let mut indptr = Vec::with_capacity(sa.len() + 1);
    let mut indices = vec![];
    let mut data = vec![];

    indptr.push(I::zero());

    let mapping = generate_ngram_index_mapping(generate_ngrams());

    let sa = sa.str().unwrap();

    for s in sa.into_iter() {
        let s = s.unwrap();
        let ngram = s.chars().ngrams(3).pad();
        let mut nnz = I::zero();
        // TODO: eventually we could use tfidf or similar
        // and add up the occurences of the ngram
        // only add ngram once
        // make sure to make match this with the csr::normlize_rows method
        for ngram_value in ngram.unique() {
            if let Some(index) = mapping.get(&ngram_value) {
                nnz += I::one();
                indices.push(*index);
                data.push(T::one());
            }
        }
        indptr.push(*indptr.last().unwrap() + nnz);
    }

    CsrMatBase::new(indptr, indices, data, sa.len(), mapping.len())
}

fn sparse_dot_topn<T, I, J>(
    a: &CsrMatBase<T, I, J>,
    b: &CsrMatBase<T, I, J>,
    ntop: usize,
) -> CsrMatBase<T, I, J>
where
    T: DataStorage,
    I: IndPtrStorage,
    J: IndexStorage,
{
    let mut indptr = Vec::with_capacity(a.rows + 1);
    let mut indices = Vec::with_capacity(a.rows * ntop);
    let mut data = Vec::with_capacity(a.rows * ntop);

    indptr.push(I::zero());

    let rows = a.rows;
    let cols = b.cols;

    let am = a.rows;
    let an = a.cols;
    let bm = b.rows;
    let bn = b.cols;

    assert_eq!(an, bm);

    for i in 0..am {
        let mut sums = vec![T::zero(); bn];
        let mut candidates = Vec::<(J, T)>::with_capacity(ntop);

        let jj_start = a.indptr[i];
        let jj_end = a.indptr[i + 1];

        for jj in jj_start.index()..jj_end.index() {
            let j = a.indices[jj];
            let v = a.data[jj];

            let kk_start = b.indptr[j.index()];
            let kk_end = b.indptr[j.index() + 1];

            for kk in kk_start.index()..kk_end.index() {
                let k = b.indices[kk];
                let w = b.data[kk];

                sums[k.index()] += v * w;
            }
        }

        let mut j = J::zero();
        for score in sums.iter() {
            if *score != T::zero() {
                candidates.push((j, *score));
            }
            j = j + J::one();
        }

        let mut nnz = I::zero();

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
            nnz += I::one();
        }

        // even if thare are no candidates (i.e. nnz=0), we push the row
        // and potentially produce a piecewise constant indptr
        indptr.push(*indptr.last().unwrap() + nnz);
    }

    CsrMatBase::new(indptr, indices, data, rows, cols)
}

fn left_parallel_sparse_dot_top_n<T, I, J>(
    a: &CsrMatBase<T, I, J>,
    b: &CsrMatBase<T, I, J>,
    ntop: usize,
    threads: usize,
) -> (Vec<(usize, usize)>, Vec<CsrMatBase<T, I, J>>)
where
    T: DataStorage,
    I: IndPtrStorage,
    J: IndexStorage,
{
    assert_eq!(a.cols, b.rows);

    let offsets = split_offsets(a.rows, threads);

    let csr_batches = offsets
        .par_iter()
        .map(|(offset, len)| {
            let a_batch = a.slice(*offset, *len);
            sparse_dot_topn(&a_batch, b, ntop)
        })
        .collect::<Vec<CsrMatBase<T, I, J>>>();

    (offsets, csr_batches)
}

fn right_parallel_sparse_dot_top_n<T, I, J>(
    a: &CsrMatBase<T, I, J>,
    b: &CsrMatBase<T, I, J>,
    ntop: usize,
    threads: usize,
) -> (Vec<(usize, usize)>, Vec<CsrMatBase<T, I, J>>)
where
    T: DataStorage,
    I: IndPtrStorage,
    J: IndexStorage,
{
    // b comes on non transposed

    assert_eq!(a.cols, b.cols);

    let offsets = split_offsets(b.rows, threads);

    let csr_batches = offsets
        .par_iter()
        .map(|(offset, len)| {
            let b_batch = b.slice(*offset, *len).transpose();
            let mut res = sparse_dot_topn(a, &b_batch, ntop);
            // after slicing and transposing we need to shift the column indices
            // and add offset to get the correct matching rows
            res.indices
                .iter_mut()
                .for_each(|j| *j = J::from_usize(j.index() + *offset));
            res
        })
        .collect::<Vec<CsrMatBase<T, I, J>>>();

    (offsets, csr_batches)
}

fn chunked_csr_to_df<T, I, J>(
    offsets: Vec<(usize, usize)>,
    csr_batches: Vec<CsrMatBase<T, I, J>>,
) -> PolarsResult<DataFrame>
where
    T: DataStorage,
    I: IndPtrStorage,
    J: IndexStorage,
{
    let mut rows: Vec<i32> = vec![];
    let mut indices: Vec<i32> = vec![];
    let mut data: Vec<f64> = vec![];

    for (offset, batch) in offsets.iter().zip(csr_batches.iter()) {
        for i in 0..batch.rows {
            let jj_start = batch.indptr[i];
            let jj_end = batch.indptr[i + 1];

            for jj in jj_start.index()..jj_end.index() {
                rows.push(i as i32 + offset.0 as i32);
                indices.push(batch.indices[jj].index() as i32);
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

fn csr_to_df<T, I, J>(csr: CsrMatBase<T, I, J>) -> PolarsResult<DataFrame>
where
    T: DataStorage,
    I: IndPtrStorage,
    J: IndexStorage,
{
    let mut rows: Vec<i32> = vec![];
    let mut indices: Vec<i32> = vec![];
    let mut data: Vec<f64> = vec![];

    for i in 0..csr.rows {
        let jj_start = csr.indptr[i];
        let jj_end = csr.indptr[i + 1];

        for jj in jj_start.index()..jj_end.index() {
            rows.push(i as i32);
            indices.push(csr.indices[jj].index() as i32);
            data.push(csr.data[jj].into());
        }
    }

    DataFrame::new(vec![
        Series::new("row".into(), rows),
        Series::new("col".into(), &indices),
        Series::new("sim".into(), data),
    ])
}

fn compute_cossim<T, I, J>(
    a: CsrMatBase<T, I, J>,
    b: CsrMatBase<T, I, J>,
    ntop: usize,
    threads: usize,
    parallelize_left: bool,
) -> PolarsResult<DataFrame>
where
    T: DataStorage,
    I: IndPtrStorage,
    J: IndexStorage,
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
            let mut a: FCsrMat = transform(sa);
            let mut b: FCsrMat = transform(sb);
            a.normalize_rows();
            b.normalize_rows();
            compute_cossim(a, b, ntop, threads, parallelize_left)
        }
        false => {
            let a: ICsrMat = transform(sa);
            let b: ICsrMat = transform(sb);
            compute_cossim(a, b, ntop, threads, parallelize_left)
        }
    }
}