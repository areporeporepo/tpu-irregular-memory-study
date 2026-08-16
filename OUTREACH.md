# Things to send, and to whom

Three messages worth sending. The first asks permission for an experiment we cannot run without
it. The other two are contributions Google can act on, which is also the best possible
introduction to the people who would answer the first.

---

## 1. Permission request: does a capacity-transfer experiment have a sanctioned form?

**To:** Google Cloud TPU support, plus the Stanford course owner of billing account
`01F867-B7EAEB-2BA7E1`, cc the faculty sponsor.
**Why it must be asked first:** the Terms of Service state a customer may not "sell, resell,
sublicense, transfer, or distribute any or all of the Services". A retroactive disclosure does not
make an unauthorised experiment authorised, and the account is the class's, not ours.

> Subject: Research question on sanctioned mechanisms for TPU capacity transfer
>
> I am a Stanford student running a measurement study of TPU v6e collective and irregular-access
> performance on project `soe-hpccenter`, published openly at
> github.com/areporeporepo/tpu-irregular-memory-study.
>
> One section of the study concerns why TPU capacity, unlike GPU capacity, has no secondary
> market. I have documented the contractual reason and I am not asking to work around it. My
> question is whether there is a *sanctioned* form of the experiment I would like to run:
> measuring what it takes for accelerator capacity to change hands, and what the transaction costs
> and reliability characteristics look like in practice.
>
> Specifically I would like to know:
> 1. Is there any programme under which a research group may make TPU capacity available to a
>    third party for measurement purposes, and if so under what terms?
> 2. Does the position change for *inference output* rather than capacity, that is, serving a
>    model on TPU and providing tokens to another party, which is not obviously resale of the
>    Service?
> 3. For the 2027 direct-hardware-sales programme, will resale or brokerage of TPU time by the
>    purchaser be permitted, restricted, or contractually prohibited as it is on Cloud today?
>
> I am happy to share all measurements from the study, including the spot preemption and capacity
> availability data, which may be independently useful to the TPU team.

**If the answer is no**, the analysis in `MARKET_MECHANISMS.md` stands on its own and the study
loses nothing. **If the answer is yes**, the experiment becomes legitimate and interesting.

---

## 2. Bug report: SparseCore Pallas is unimplemented on v6e while documented as supported

**To:** the JAX repository, `jax-ml/jax`.

The documentation for SparseCore kernel writing lists v4, v5p, v6e and v7x as having SparseCores,
and uses a 7x system for all examples. On v6e with JAX 0.10.2:

- `jax.experimental.pallas.tpu_sc` imports cleanly
- `plsc.get_sparse_core_info()` reports `num_cores=2, num_subcores=16, num_lanes=8,
  vmem_capacity_bytes=262144, dma_granule_size_bytes=32`
- the documented gather kernel fails to compile at every shape tried, with one of:
  `'tpu.enqueue_indirect_dma' op Not implemented`, `'sc_tpu.enqueue_transfer' op Not implemented`,
  or `'memref.alloca' op E3000: CompileTimeSparseCore...`

Seven variants were tried, spanning row widths 8 to 256, index counts 1K to 8K, window sizes 8 to
1024, and both float32 and bfloat16. A minimal reproduction is `sc_debug.py` in the repository.

The useful outcome is a documentation change if v6e is genuinely unsupported, since the hardware
reporting its SparseCore configuration strongly implies otherwise.

---

## 3. Bug report: XLA RET_CHECK on sub-communicator all_reduce

**To:** `openxla/xla`, or JAX if it reproduces at that level.

On a v6e-256 slice (64 processes), `jax.lax.psum` under `shard_map` fails with
`INTERNAL: RET_CHECK failure` for 8-chip and 16-chip sub-meshes, 8 of 48 configurations, while
`all_to_all` succeeds on exactly the same meshes and the same `psum` configurations succeed on a
v6e-32 slice. It also reproduces on a v6e-32 for a strided 8-chip subset.

This matters beyond the crash: partitioning a slice between jobs is ordinary practice, and a
collective that works on the whole slice but fails on a subset of it is a sharp edge.

---

## Why send 2 and 3 first

They cost Google nothing to receive and are immediately actionable, which is a better opening than
a permissions question. They also establish that the study found real things, which is the only
credential that matters when asking an engineering team for anything.
