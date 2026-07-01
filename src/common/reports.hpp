// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
//
// Report writers (append-mode CSV + JSON) and console summaries, mirroring
// tests/lib/reports.py column orders and field names exactly.
#pragma once

#include <string>
#include <vector>

#include "common/metrics.hpp"

namespace mb {

void write_gen_csv(const std::string& path, const std::vector<GenBenchmarkResult>& results);
void write_gen_json(const std::string& path, const std::vector<GenBenchmarkResult>& results);
void print_gen_summary(const GenBenchmarkResult& r);

void write_bert_csv(const std::string& path, const std::vector<BertBenchmarkResult>& results);
void write_bert_json(const std::string& path, const std::vector<BertBenchmarkResult>& results);
void print_bert_summary(const BertBenchmarkResult& r);

void write_embed_csv(const std::string& path, const std::vector<EmbedBenchmarkResult>& results);
void write_embed_json(const std::string& path, const std::vector<EmbedBenchmarkResult>& results);
void print_embed_summary(const EmbedBenchmarkResult& r);

void write_rerank_csv(const std::string& path, const std::vector<RerankBenchmarkResult>& results);
void write_rerank_json(const std::string& path, const std::vector<RerankBenchmarkResult>& results);
void print_rerank_summary(const RerankBenchmarkResult& r);

}  // namespace mb
