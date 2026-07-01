// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
//
// model_bench — OpenVINO GenAI C++ benchmark suite dispatcher.
#include <cstdio>
#include <exception>
#include <string>

#include "common/cli.hpp"
#include "tasks/tasks.hpp"

int main(int argc, char** argv) {
    try {
        mb::Args args = mb::parse_args(argc, argv);
        if (args.help || args.task.empty()) {
            mb::print_usage(args.task);
            return args.task.empty() && argc < 2 ? 1 : 0;
        }
        if (args.task == "llm") return mb::run_llm(args);
        if (args.task == "vlm") return mb::run_vlm(args);
        if (args.task == "lingua") return mb::run_lingua(args);
        if (args.task == "embedding") return mb::run_embedding(args);
        if (args.task == "reranker") return mb::run_reranker(args);
        std::fprintf(stderr, "ERROR: unknown task '%s'\n\n", args.task.c_str());
        mb::print_usage("");
        return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
