//! Whole-genome domain calling: the TADLib `domaincaller.Genome` pipeline
//! (`tadlib/domaincaller/genomeLev.py`). The flow is:
//!
//! 1. [`Genome::new`] loads every non-excluded chromosome as a [`Chrom`]
//!    from the balance-column-corrected upper-triangle matrix;
//! 2. [`Genome::learning`] computes per-chromosome DI splits and trains
//!    **one shared HMM** (Baum-Welch from `oriHMMParams`) on the non-zero
//!    segments of all chromosomes;
//! 3. [`Genome::call_domains`] decodes each chromosome with that shared HMM
//!    and merges the per-chromosome domains, dropping noise > 0.5;
//! 4. [`Genome::di_track`] emits the CLI-format DI bedGraph rows
//!    (chrom, start, end, DI), last bin clipped to the chromosome size.
//!
//! The reference output of this pipeline (run inside a TADLib container) is
//! the `tests/data/4DNFIZ1ZVXC8.50kb.tadlib.{dis,domains}` pair the
//! `tests/domaincaller.rs` regression compares against.

use std::rc::Rc;

use crate::cooler::Cooler;
use crate::stats::HiddenMarkovModel;

use super::chrom::Chrom;

/// Whole-genome `domaincaller` context with a single shared HMM.
pub struct Genome {
    /// Bin size in base pairs.
    pub res: u64,
    /// Included chromosomes, in cooler file order.
    pub chroms: Vec<String>,
    /// Size of each included chromosome in base pairs (parallels `chroms`).
    pub chrom_sizes: Vec<u64>,
    data: Vec<Chrom>,
    hmm: Option<Rc<HiddenMarkovModel>>,
}

impl Genome {
    /// Load a cooler at `res` resolution. `balance` names a `bins` column
    /// (e.g. `"weight"`) applied like cooler-python's `matrix(balance=...)`;
    /// `exclude` chromosomes (e.g. `["chr4", "chrY", "chrM"]`) are skipped.
    pub fn new(
        cool: &Cooler,
        res: u64,
        balance: Option<&str>,
        exclude: &[&str],
    ) -> crate::Result<Self> {
        let chroms = cool.chroms()?;
        let weight = match balance {
            Some(name) => cool.bins_column_f64(name)?.ok_or_else(|| {
                crate::Error::InvalidInput(format!("no 'bins/{name}' column for balance"))
            })?,
            None => Vec::new(),
        };

        let mut data = Vec::new();
        let mut included = Vec::new();
        let mut sizes = Vec::new();
        let mut bin_off: i64 = 0;
        for ch in &chroms {
            // number of bins covering `ch.length` at `res`
            let n = ((ch.length as u64).div_ceil(res)) as usize;
            if !exclude.contains(&ch.name.as_str()) {
                // Upper-triangle entries in local bin coordinates, exactly as
                // TADLib's `triu(matrix(balance=..., sparse=True).fetch(c))`.
                let mut entries = Vec::new();
                for p in cool.pixels_for_bins(bin_off, bin_off + n as i64)? {
                    let (b1, b2) = (p.bin1_id as usize, p.bin2_id as usize);
                    // drop pixels whose bin2 falls beyond the chromosome and
                    // the lower triangle (bin1 > bin2), as the CLI does
                    if b2 >= (bin_off + n as i64) as usize || b1 > b2 {
                        continue;
                    }
                    let mut v = p.count;
                    if !weight.is_empty() {
                        v *= weight[b1] * weight[b2];
                    }
                    entries.push((b1 - bin_off as usize, b2 - bin_off as usize, v));
                }
                data.push(Chrom::new(&ch.name, res, n, &entries));
                included.push(ch.name.clone());
                sizes.push(ch.length as u64);
            }
            bin_off += n as i64;
        }

        Ok(Genome {
            res,
            chroms: included,
            chrom_sizes: sizes,
            data,
            hmm: None,
        })
    }

    /// TADLib `Genome.learning`: per-chromosome DI splits, then a single
    /// Baum-Welch fit (one HMM shared by every chromosome).
    pub fn learning(&mut self) {
        let mut seqs = Vec::new();
        for c in &mut self.data {
            c.compute_di();
            seqs.extend(c.train_data());
        }
        self.hmm = Some(Rc::new(Chrom::train_hmm(&seqs)));
    }

    /// TADLib `Genome.callDomains` + `outputDomain`: decode each chromosome
    /// with the shared HMM and return merged `(chrom, start_bp, end_bp)`
    /// domains in file order, dropping noise > 0.5. Requires
    /// [`Genome::learning`] first.
    pub fn call_domains(&mut self) -> Vec<(String, u64, u64)> {
        let hmm = self
            .hmm
            .as_ref()
            .expect("Genome::learning() must run before Genome::call_domains()");
        let mut out = Vec::new();
        for (c, name) in self.data.iter_mut().zip(&self.chroms) {
            c.set_hmm(Rc::clone(hmm));
            c.call_with_hmm();
            for d in &c.domains {
                if d[2] > 0.5 {
                    continue;
                }
                out.push((name.clone(), d[0] as u64, d[1] as u64));
            }
        }
        out
    }

    /// CLI-format DI bedGraph rows `(chrom, start, end, DI)` for the whole
    /// genome in file order; the final bin of each chromosome is clipped to
    /// its size.
    pub fn di_track(&self) -> Vec<(String, u64, u64, f64)> {
        let mut out = Vec::new();
        for (c, (name, size)) in self
            .data
            .iter()
            .zip(self.chroms.iter().zip(self.chrom_sizes.iter()))
        {
            for (i, &di) in c.dis.iter().enumerate() {
                let start = (i as u64) * self.res;
                let end = (((i as u64) + 1) * self.res).min(*size);
                out.push((name.clone(), start, end, di));
            }
        }
        out
    }
}
