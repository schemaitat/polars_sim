use crate::csr::*;
use crate::helper::split_offsets;
use itertools::{iproduct, Itertools};
use lazy_static::lazy_static;
use ngrams::Ngram;
use polars::prelude::*;
use polars_core::utils::accumulate_dataframes_vertical;
use rayon::prelude::*;
use std::collections::HashMap;

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

fn normalize_string(s: &str) -> String {
    s.to_lowercase()
}

fn transform(sa: &Series) -> Csr {
    let mut indptr = vec![0];
    let mut indices = vec![];
    let mut data = vec![];

    let sa = sa.str().unwrap();

    for s in sa.into_iter() {
        let s = normalize_string(s.unwrap());
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
                data.push(1.0);
            }
        }
        indptr.push(indptr.last().unwrap() + nnz);
    }

    Csr {
        indptr,
        indices,
        data,
        rows: sa.len(),
        cols: MAPPING.len(),
    }
}

fn sparse_dot_topn(a: &Csr, b: &Csr, ntop: usize) -> Csr {
    let mut indptr = vec![0];
    let mut indices = vec![];
    let mut data = vec![];

    let rows = a.rows;
    let cols = b.cols;

    let am = a.rows;
    let an = a.cols;
    let bm = b.rows;
    let bn = b.cols;

    assert_eq!(an, bm);

    for i in 0..am {
        let mut sums = vec![0 as f64; bn];
        let mut candidates = vec![];

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
            if *score > 0.0 {
                candidates.push((j, *score));
            }
        }

        let mut nnz = 0;

        let topn_candidates = if ntop < candidates.len() {
            candidates
                .select_nth_unstable_by(ntop, |a, b| b.1.partial_cmp(&a.1).unwrap())
                .0
                .to_vec()
        } else {
            candidates
        };

        for (j, v) in topn_candidates {
            indices.push(j);
            data.push(v);
            nnz += 1;
        }

        indptr.push(indptr.last().unwrap() + nnz);
    }

    Csr {
        indptr,
        indices,
        data,
        rows,
        cols,
    }
}

fn compute_cossim(
    sa: &Series,
    sb: &Series,
    ntop: usize,
    row_offset: usize,
    normalize: bool,
) -> PolarsResult<DataFrame> {
    let mut a = transform(sa);
    let mut b = transform(sb);

    if normalize {
        normalize_rows(&mut a);
        normalize_rows(&mut b);
    }

    let c = sparse_dot_topn(&a, &transpose_csr(&b), ntop);

    let mut rows = vec![];
    let mut indices = vec![];
    let mut data = vec![];

    for i in 0..c.rows {
        let jj_start = c.indptr[i];
        let jj_end = c.indptr[i + 1];

        for jj in jj_start..jj_end {
            rows.push((i + row_offset) as i64);
            indices.push(c.indices[jj] as i64);
            data.push(c.data[jj] as f64);
        }
    }

    DataFrame::new(vec![
        Series::new("row".into(), rows),
        Series::new("col".into(), indices),
        Series::new("sim".into(), data),
    ])
}

pub(super) fn awesome_cossim(
    df_left: DataFrame,
    df_right: DataFrame,
    col_left: &str,
    col_right: &str,
    ntop: usize,
    threads: Option<usize>,
    normalize: Option<bool>,
) -> PolarsResult<DataFrame> {
    let threads = threads.unwrap_or(rayon::current_num_threads());

    let normalize = normalize.unwrap_or(false);

    if threads > 1 {
        let offsets = split_offsets(df_left.height(), threads);

        let dfs = offsets
            .par_iter()
            .map(|(offset, len)| {
                let sa = df_left
                    .column(col_left)
                    .unwrap()
                    .slice(*offset as i64, *len);
                let sb = df_right.column(col_right).unwrap();
                compute_cossim(&sa, sb, ntop, *offset, normalize)
            })
            .collect::<PolarsResult<Vec<_>>>()?;
        Ok(accumulate_dataframes_vertical(dfs)?)
    } else {
        let sa = df_left.column(col_left).unwrap();
        let sb = df_right.column(col_right).unwrap();
        compute_cossim(sa, sb, ntop, 0, normalize)
    }
}
