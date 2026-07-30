# Use global base if possible
ifndef MILABENCH_BASE
	MILABENCH_BASE="base"
endif

export MILABENCH_BASE

BENCH_NAME=torchsrun
MILABENCH_CONFIG=dev.yaml
MILABENCH_ARGS=--config $(MILABENCH_CONFIG) --base $(MILABENCH_BASE)

all: install prepare nodes

install:
	milabench install $(MILABENCH_ARGS) --force

prepare:
	milabench prepare $(MILABENCH_ARGS)

nodes:
	milabench run $(MILABENCH_ARGS) --select $(BENCH_NAME)

many:
	milabench run $(MILABENCH_ARGS) --select $(BENCH_NAME)-nodes
