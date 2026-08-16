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

---

## 4. Access request: container.admin and billing.viewer

**To:** `smjones@stanford.edu` (project owner, technical contact, and the requester of record)
**Cc:** `tbrooke@stanford.edu` (financial contact, technical contact 2)
**Note:** those addresses are inferred from the project labels `su-owner_name=su__smjones` and
`su-fin_contact=su__tbrooke`. Confirm them before sending.

**Why this cannot be self-served:** Privileged Access Manager is not configured on this project, and
`gcloud projects add-iam-policy-binding` returns `does not have permission`. There is no request
API. A human with admin rights has to run two commands.

> Subject: Access request on soe-hpccenter: container.admin and billing.viewer
>
> Hi — I am running a TPU measurement study on `soe-hpccenter`, published openly at
> github.com/areporeporepo/tpu-irregular-memory-study. Two access grants would help, and one of
> them I think helps you more than it helps me.
>
> **1. `roles/billing.viewer` on billing account `01F867-B7EAEB-2BA7E1`.** This is the important
> one. At the moment nobody using the project can see how much credit remains, so we are all
> estimating spend from instance uptime and the published spot rate. I have a script that does this
> and deliberately over-bills to stay safe, but a real balance would replace guesswork for everyone
> on the project, not just me.
>
> ```
> gcloud billing accounts add-iam-policy-binding 01F867-B7EAEB-2BA7E1 \
>   --member="user:qanh@stanford.edu" --role="roles/billing.viewer"
> ```
>
> **2. `roles/container.admin` on `soe-hpccenter`.** The four Autopilot clusters currently have no
> student namespaces, no ResourceQuotas and no Kueue, so the first person to claim capacity holds it
> until they release it. I would like to set up per-student namespaces with hard TPU quotas and a
> Kueue cohort so idle quota is borrowed rather than wasted, which is what lets hardware sized for
> eight serve a class of thirty. I will hand over the manifests either way.
>
> ```
> gcloud projects add-iam-policy-binding soe-hpccenter \
>   --member="user:qanh@stanford.edu" --role="roles/container.admin"
> ```
>
> **If only one is possible, `billing.viewer` is the one worth granting.** It is read-only, it
> cannot affect anyone's workloads, and it answers a question the whole project currently cannot
> answer. `container.admin` is a larger grant and I am happy to instead send you the manifests to
> apply yourself, or to work in a single namespace you create for me.
>
> Two findings from this week that may be useful regardless of the answer:
>
> - `tpu-inference` 0.26.0 is uninstallable: it pins `libtpu==0.0.43`, a version never published to
>   PyPI. The 0.25.0 pair works. Students will lose an afternoon to this.
> - v6e capacity is exhausted in every us-east5 zone and available in us-east1-d and
>   asia-northeast1-b. On-demand does not help; it is refused with the same capacity error. But
>   **v5p is available in us-east5-a**, and at 95 GB per chip it is the better choice for
>   fine-tuning anyway.

**What each grant unblocks, concretely:**

| grant | unblocks |
|---|---|
| `roles/billing.viewer` | the real credit balance and burn rate, replacing `budget_guard.py`'s estimate |
| `roles/container.admin` | namespaces, ResourceQuotas, Kueue cohorts, taints; multi-tenant fairness for a class |
| neither | everything in this study continues; both are conveniences, not blockers |
