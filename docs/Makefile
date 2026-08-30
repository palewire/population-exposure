# Minimal makefile for Sphinx documentation

SPHINXOPTS ?=
UV ?= uv
RUN_DOCS ?= $(if $(UV_PYTHON),UV_PYTHON=$(UV_PYTHON)) $(UV) run --directory .. --group docs
SPHINXBUILD ?= $(RUN_DOCS) sphinx-build
SOURCEDIR = docs
BUILDDIR = docs/_build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile livehtml

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

livehtml:
	$(RUN_DOCS) sphinx-autobuild -b html $(SOURCEDIR) $(BUILDDIR)/html
