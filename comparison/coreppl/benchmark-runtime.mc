-- Shared reporting outside the inferred model. No per-particle printing.
include "seq.mc"
include "string.mc"
include "math.mc"
include "ext/dist-ext.mc"
include "ext/math-ext.mc"

-- float2string prints only about 12 significant digits in this runtime.
-- Emit a high part and a residual so the Python reader can recover Float
-- precision without changing the compiler, runtime, or inference arithmetic.
let preciseFloat = lam x.
  let high = float2string x in
  let low = float2string (subf x (string2float high)) in
  strJoin ":" [high, low]

let printVector = lam values.
  print (strJoin " " (map preciseFloat values)); print "\n"

-- Each particle contributes a conditional mean and conditional variance.
-- Samples are the special case with zero conditional variance.
let reportMoments = lam elapsedMs. lam logZ. lam samples. lam conditionalVariances. lam logWeights.
  let summaryStart = wallTimeMs () in
  let weights = map exp logWeights in
  let weightSum = foldl addf 0.0 weights in
  let weights = map (lam w. divf w weightSum) weights in
  let zeros = create (length (head samples)) (lam. 0.0) in
  -- Center the sums so identical particles give exactly identical moments
  -- independently of particle count, rather than accumulating rounding error.
  let origin = head samples in
  let offsets = foldl2 (lam acc. lam xs. lam w.
    zipWith (lam a. lam delta. addf a (mulf w delta)) acc (zipWith subf xs origin)
  ) zeros samples weights in
  let means = zipWith addf origin offsets in
  let between = foldl2 (lam acc. lam xs. lam w.
    zipWith (lam a. lam pair.
      match pair with (x, mu) in
      let delta = subf x mu in
      addf a (mulf w (mulf delta delta))
    ) acc (zip xs means)
  ) zeros samples weights in
  let varianceOrigin = head conditionalVariances in
  let varianceOffsets = foldl2 (lam acc. lam vs. lam w.
    zipWith (lam a. lam delta. addf a (mulf w delta)) acc (zipWith subf vs varianceOrigin)
  ) zeros conditionalVariances weights in
  let within = zipWith addf varianceOrigin varianceOffsets in
  let variances = zipWith addf within between in
  let ess = divf 1.0 (foldl (lam acc. lam w. addf acc (mulf w w)) 0.0 weights) in
  let summaryMs = subf (wallTimeMs ()) summaryStart in
  printVector [elapsedMs, summaryMs, logZ, ess];
  printVector means;
  printVector variances

let report = lam elapsedMs. lam logZ. lam samples. lam logWeights.
  let zeros = create (length (head samples)) (lam. 0.0) in
  reportMoments elapsedMs logZ samples (map (lam. zeros) samples) logWeights
