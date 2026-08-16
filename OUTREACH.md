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

## 2. Bug report: the SparseCore gather kernel compiles at only some shapes, and the failure says nothing useful

**To:** the JAX repository, `jax-ml/jax`.

An earlier draft of this section claimed the kernel was unimplemented on v6e. That was wrong, and
running the same script on two generations is what showed it: one shape does compile on v6e. The
real problem is narrower and more actionable.

The documented gather kernel, taken from the SparseCore kernel-writing guide and changed only in its
shapes, was compiled at 16 configurations on each of two chips with JAX 0.10.2:

| chip | SparseCore configuration reported | compiled |
|---|---|---|
| v5p | `num_cores=4, num_subcores=16, num_lanes=8, vmem_capacity_bytes=524288` | 9 of 16 |
| v6e | `num_cores=2, num_subcores=16, num_lanes=8, vmem_capacity_bytes=262144` | 1 of 16 |

Every shape that compiled was bit-exact against `jnp.take`, so the kernel itself is right. The
failures all report `INTERNAL: Failed to run MLO pass pipeline` at one of three source locations,
with no indication of which constraint was violated.

The envelope has no documented rule that we could find. On v5p, row widths of 128 and 256 compile
and 8, 32 and 512 do not; windows of 128 and 256 compile and 64 and 512 do not; and a bfloat16
configuration fails at a shape whose float32 twin succeeds. Reproduction is
`experiment9_sparsecore_v5p.py` in the repository, which prints the whole grid.

Either of two outcomes is useful: a documented constraint on shapes, or a compile error that names
the constraint instead of reporting an internal pass failure. The second matters more, because a
user meeting this today cannot tell an unsupported shape from a bug.

**Why it is worth Google's attention beyond tidiness.** On the shapes that do compile, the
SparseCore is 2.3x faster than the TensorCore for gathers out of tables larger than the compiler's
on-chip promotion budget, which is every embedding table and KV cache of a real size. The narrow
envelope is standing between users and that speedup.

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

## 3b. Question for the XLA team: is the gather promotion budget documented, and is it tunable?

**To:** `openxla/xla`, as a question rather than a bug, because the behaviour may well be intended.

A gather's source table is promoted to memory space `S(1)` when it fits inside a fixed budget, and
left in HBM when it does not. The performance difference is large and the transition is sharp:

| chip | last size promoted | reads as | promoted | in HBM | step |
|---|---|---|---|---|---|
| v5p | 50,266,112 B | 48 MiB &minus; 64 KiB | 256 GB/s | 77 GB/s | 3.33x |
| v6e | 100,597,760 B | 96 MiB &minus; 64 KiB | 154 GB/s | 42 GB/s | 3.64x |

Measured with 16384 indices into a table of 128-wide float32 rows, 32 gathers chained inside one
jit so host dispatch is amortised. The threshold is identical at 128-, 256- and 512-wide rows, so it
is a byte budget; at 64-wide rows it halves, consistent with the budget counting bytes allocated
after padding to `T(8,128)`. It does not move when the index count changes by 64x, so the index
vector is not drawn from the same allowance.

Three questions:

1. **Is the budget documented anywhere?** We could not find it, and a 3.3x cliff at a size no user
   chose is the kind of thing worth a paragraph in the performance guide.
2. **Where does the 64 KiB come from?** Both numbers are exactly 64 KiB below a round power of two.
   We assumed it was the index vector, tested that, and it was not.
3. **Is it tunable?** If `xla_tpu_scoped_vmem_limit_kib` or a neighbour moves this threshold, then
   any model whose embedding table sits just above it has a one-flag 3x win available, and that
   should be written down. If it is not tunable, knowing that is equally useful, because then the
   only remedy is a SparseCore kernel.

**Why this is worth someone's time.** Every embedding table and every KV cache of a realistic size
is far above the budget, so production gathers are always on the slow path, and the usual advice for
irregular access does not help: we measured that sorting the indices and grouping them into
consecutive blocks change the rate by less than 0.2%. Meanwhile an A100 running the same gather is
flat in table size to within 1.06x. The TPU's disadvantage here is a placement decision, not a
memory system, which is why it seems worth asking about rather than working around.

Full method, data and reproduction: github.com/areporeporepo/tpu-irregular-memory-study, and
`hlo_gather_lowering.py --bisect` reproduces the threshold in under a minute using only compiles.

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
