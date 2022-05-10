#! /usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
## Copyright (c) 2022 Jeet Sukumaran.
## All rights reserved.
##
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are met:
##
##     * Redistributions of source code must retain the above copyright
##       notice, this list of conditions and the following disclaimer.
##     * Redistributions in binary form must reproduce the above copyright
##       notice, this list of conditions and the following disclaimer in the
##       documentation and/or other materials provided with the distribution.
##     * The names of its contributors may not be used to endorse or promote
##       products derived from this software without specific prior written
##       permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
## ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
## WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
## DISCLAIMED. IN NO EVENT SHALL JEET SUKUMARAN BE LIABLE FOR ANY DIRECT,
## INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
## BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
## DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
## LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
## OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
## ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
##
##############################################################################

import os
import pathlib
import sys
import argparse
import functools
import pathlib
import random
import subprocess
from dendropy.interop import seqgen
import dendropy

def _log(
    message,
    level=0,
    *,
    indent_size=2,
    out=sys.stderr,
    bullet_fn=None,
):
    if bullet_fn is None:
        bullet_fn = lambda x: "-"
    parts = []
    spacer = (" " * (indent_size)) * level
    parts.append(spacer)
    bullet = bullet_fn(level)
    spacer = " " * (indent_size-1)
    parts.append(f"{bullet}{spacer}")
    leader = "".join(parts)
    out.write(f"{leader}{message}\n")

def _get_rng(seed=None):
    if seed is None:
        seed = random.getrandbits(32)
    _log(f"Using random seed: {seed}")
    rng = random.Random(seed)
    return rng

class MsIslandModel:

    def __init__(
        self,
        rng,
        output_prefix,
        n_demes,
        migration_rate,
        # fragmentation_age,
        mutation_rate=1e-4,
        n_samples_per_pop=4,
        sequence_length=1000000,
        population_size=10000,
        ploidy=1,
    ):
        self.output_prefix = output_prefix
        self.rng = rng
        self.n_demes = n_demes
        self.migration_rate = migration_rate
        # self.fragmentation_age = fragmentation_age
        # self.fragmentation_time_t = fragmentation_age * population_size
        self.mutation_rate = mutation_rate
        self.n_samples_per_pop = n_samples_per_pop
        self.sequence_length = sequence_length
        self.population_size = population_size
        self.ploidy = ploidy
        self.output_column_delimiter = "\t"
        self.index_labels = {
            "population": "idx_population",
        }
        self.seq_gen = seqgen.SeqGen(rng=self.rng)

    def report_setup(self):
        _log(f"Islands: {self.n_demes}")
        _log(f"Migration Rate: {self.migration_rate}")
        _log(f"Mutation Rate: {self.mutation_rate}")
        _log(f"Population Size: {self.population_size}")
        theta = 4 * self.population_size * self.mutation_rate
        _log(f"Theta: {theta}")
        _log(f"Sequence Length: {self.sequence_length}")
        # _log(f"Fragmentation Age (N): {self.fragmentation_age}")
        # _log(f"Fragmentation Age (t): {self.fragmentation_time_t}")

    def run(
        self,
        n_replicates,
    ):
        self.report_setup()
        for rep_idx in range(n_replicates):
            output_path = f"{self.output_prefix}_{rep_idx+1:00d}.fasta"
            tree = self.run_ms_generate_tree()
            raw_fasta = self.seq_gen.generate_raw(
                tree,
                output_format="fasta",
                tree_serialization_kwargs={
                    "unquoted_underscores": True,
                },
            )

            out_path = f"{self.output_prefix}_{rep_idx+1:04d}.fasta"
            with open(out_path, "w") as dest:
                dest.write(raw_fasta)

    def execute_command(
        self,
        cmd,
        **kwargs,
    ):
        cmd = [str(v) for v in cmd]
        _log(f"Executing: {' '.join(cmd)}")
        cp = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            **kwargs,
        )
        return cp

    def run_ms_generate_tree(self):
        """
        Generates a sample tree, with branch lengths in units of theta.
        """
        ms_cmd = ["ms"]
        ms_cmd.append(self.n_samples_per_pop * self.n_demes)
        ms_cmd.append(1) # n_replicates
        ms_cmd.append("-T")
        ms_cmd.append("-seeds")
        for _ in range(3):
            ms_cmd.append(str(self.rng.getrandbits(32)))
        ms_cmd.append("-I")
        ms_cmd.append(self.n_demes)
        sample_population_idx_map = {}
        sample_idx = 0
        for pop_idx in range(self.n_demes):
            ms_cmd.append(self.n_samples_per_pop)
            for sidx in range(self.n_samples_per_pop):
                sample_population_idx_map[str(sample_idx+1)] = pop_idx+1
                sample_idx += 1
        ms_cmd.append(self.migration_rate)
        stdout = self.execute_command(ms_cmd).stdout.decode("utf-8")
        results = []
        is_capture_next = False
        for row in stdout.split("\n"):
            row = row.strip()
            if not row:
                is_capture_next = False
            if is_capture_next:
                results.append(row)
            elif row.startswith("//"):
                is_capture_next = True
        assert len(results) == 1
        tree_str = results[0]
        tree = dendropy.Tree.get(
            data=tree_str,
            schema="newick",
            rooting="force-rooted",
        )
        tree.tree_str = tree_str
        tree.sample_population_idx_map = sample_population_idx_map
        tree.sample_population_label_map = {}
        for nd in tree:
            if nd.edge.length is None:
                nd.edge.length = 0.0
            nd.edge.length_in_theta = nd.edge.length
            nd.edge.length = nd.edge.length_in_theta / (4 * self.population_size)
            if nd.is_leaf():
                nd.taxon.population_idx = sample_population_idx_map[nd.taxon.label]
                nd.taxon.label = f"P{nd.taxon.population_idx:03d}_i{nd.taxon.label}"
                tree.sample_population_label_map[nd.taxon.label] = nd.taxon.population_idx
        return tree

    # def generate_data_frame_from_fasta(
    #     self,
    #     fasta_text,
    #     sample_population_label_map,
    # ):
    #     data_rows = []
    #     label_column = []
    #     current_row = None
    #     for text_row in fasta_text.split("\n"):
    #         text_row = text_row.strip()
    #         if not text_row:
    #             continue
    #         if text_row.startswith(">"):
    #             if current_row:
    #                 data_rows.append(current_row)
    #                 assert len(label_column) == len(data_rows)
    #             current_row = []
    #             sample_label = text_row[1:].replace("'", "")
    #             population_label = sample_population_label_map[sample_label]
    #             label_column.append(population_label)
    #             current_row.append(population_label)
    #         else:
    #             for site_char in text_row:
    #                 site_char = site_char.upper()
    #                 ch_idx = self.site_character_index_map_fn(site_char)
    #                 current_row.append(ch_idx)
    #     gdf = calculate.GenomicDataFrame(
    #         label_series=[ pd.Series(label_column) ],
    #         sequence_array=data_rows,
    #     )
    #     # df_seqs = pd.DataFrame(data_rows)
    #     # df = pd.concat(
    #     #     (pd.Series(label_column), df_seqs),
    #     #     axis=1,
    #     # )
    #     return gdf

    # def generate_data_frame(self):
    #     tree = self.run_ms_generate_tree()
    #     raw_fasta = self.seq_gen.generate_raw(
    #         tree,
    #         output_format="fasta",
    #         tree_serialization_kwargs={
    #             "unquoted_underscores": True,
    #         }
    #     )
    #     gdf = calculate.GenomicDataFrame.from_fasta(
    #         fasta_text=raw_fasta,
    #         label_generating_fns={
    #             self.index_labels["population"]: lambda x: tree.sample_population_label_map[x],
    #         }
    #     )
    #     g_stats = gdf.calc_stats(
    #         group_label_column_name=self.index_labels["population"],
    #     )
    #     _log(f"(local) Diversity:\n {g_stats.diversity}")
    #     _log(f"(local) Divergence:\n {g_stats.divergence}")
    #     _log(f"(local) Fst:\n {g_stats.fst}")
    #     for key in g_stats.pg_fst:
    #         _log(f"(local) popGenStat {key}:\n {g_stats.pg_fst[key]}")
    #     # assert False
    #     df = gdf.as_merged_df()
    #     return df

def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        "-o", "--output-prefix",
        action="store",
        default="data",
        help="Prefix for output files [default=%(default)s].")
    parser.add_argument(
        "-k", "--num-islands",
        dest="n_demes",
        action="store",
        type=int,
        default=4,
        help="Number of 'islands' or subpopulations [%(default)s].")
    parser.add_argument(
        "-n", "--num-samples-per-pop",
        dest="n_samples_per_pop",
        action="store",
        type=int,
        default=4,
        help="Number of 'islands' or subpopulations [%(default)s].")
    parser.add_argument(
        "-t", "--fragmentation-time",
        dest="time_of_fragmentation",
        action="store",
        type=float,
        default=20,
        help="Time (in the past) since the populations fragmented, in units of N. [default=%(default)s]")
    parser.add_argument(
        "-m", "--migration-rate",
        action="store",
        type=float,
        help="Migration rate (m_ij).")
    parser.add_argument(
        "-u", "--mutation-rate",
        action="store",
        type=float,
        default=1e-8,
        help="Mutation rate [default=%(default)s].")
    parser.add_argument(
        "-N", "--population-size",
        action="store",
        type=float,
        default=1.0,
        help="Size of each of the island subpopulations [default=%(default)s].")
    parser.add_argument(
        "-l", "--sequence-length",
        action="store",
        type=float,
        default=10000,
        help="Number of sites [default=%(default)s].")
    parser.add_argument(
        "-z",
        "--random-seed",
        metavar="SEED",
        type=int,
        help="Random number seed.",
    )
    parser.add_argument(
        "-r",
        "--num-replicates",
        dest="n_replicates",
        type=int,
        default=1,
        help="Number of replicates. [default = %(default)s].",
    )
    args = parser.parse_args()
    if args.migration_rate is None:
        sys.exit("Please specify migration rate using '-m'/'--migration-rate'")
    rng = _get_rng(seed=args.random_seed)
    effective_migration_rate = args.migration_rate
    # if args.simulator_implementation == "ms":
    #     generator_type = MsIslandModel
    # elif args.simulator_implementation == "msprime":
    #     generator_type = MsprimeIslandModel
    # else:
    #     raise ValueError(args.simulator_implementation)
    generator_type = MsIslandModel
    island_model = generator_type(
        n_demes=args.n_demes,
        n_samples_per_pop=args.n_samples_per_pop,
        output_prefix=args.output_prefix,
        # fragmentation_age=args.time_of_fragmentation,
        migration_rate=effective_migration_rate,
        mutation_rate=args.mutation_rate,
        population_size=args.population_size,
        sequence_length=args.sequence_length,
        rng=rng,
    )
    island_model.run(n_replicates=args.n_replicates)

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()
