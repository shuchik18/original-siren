"""Independent analytic checks and an executable check of the CorePPL shim.

Run with: python3 -B -m unittest -v test_benchmarks.py
"""

import math
from pathlib import Path
import re
import subprocess
import unittest
from fractions import Fraction
from itertools import product

from benchmark import HERE, REPO, compiler_environment, error_metrics, parse_coreppl, summarize, wrapper_source
from posterior_readout import adapt_terminal_draw, adapt_final_scalar_fold, adapt_posterior_readout
from reference import MODELS, WHEELS_DATA, OUTLIER_DATA, SLAM_DATA, condition, gaussian_problem, reference
from run_experiments import comparison_row


def batch_condition(mean, covariance, observations):
    """Independent batch Gaussian conditioning via Cholesky solves."""
    hs = [h for _, h, _ in observations]
    cross = [[sum(row[k] * h[k] for k in range(len(mean))) for h in hs]
             for row in covariance]
    s = [[sum(h[k] * cross[k][j] for k in range(len(mean)))
          + (observations[i][2] if i == j else 0.0)
          for j in range(len(hs))] for i, h in enumerate(hs)]
    n = len(hs)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            value = s[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            lower[i][j] = math.sqrt(value) if i == j else value / lower[j][j]

    def solve(b):
        y, x = [0.0] * n, [0.0] * n
        for i in range(n):
            y[i] = (b[i] - sum(lower[i][k] * y[k] for k in range(i))) / lower[i][i]
        for i in reversed(range(n)):
            x[i] = (y[i] - sum(lower[k][i] * x[k] for k in range(i+1, n))) / lower[i][i]
        return x

    residual = [y - sum(hk * mk for hk, mk in zip(h, mean)) for y, h, _ in observations]
    alpha = solve(residual)
    posterior = [m + sum(c*a for c, a in zip(row, alpha)) for m, row in zip(mean, cross)]
    variance = [covariance[i][i] - sum(c*x for c, x in zip(row, solve(row)))
                for i, row in enumerate(cross)]
    log_z = -0.5 * (n * math.log(2*math.pi) + 2*sum(math.log(lower[i][i]) for i in range(n))
                    + sum(r*a for r, a in zip(residual, alpha)))
    return posterior, variance, log_z


class ReferenceTests(unittest.TestCase):
    def test_gaussian_hybrid_against_direct_quadrature(self):
        # Independently integrate prior(x) * [0.5*p(0|x,b=true) +
        # 0.5*p(0|b=false)] over x; no conjugate or branch posterior formulas.
        def normal(x, variance):
            return math.exp(-x*x / (2*variance)) / math.sqrt(2*math.pi*variance)

        truth = reference("gaussian-gaussian-hybrid")
        for intervals in (4096, 8192):
            # ±40 is >12 prior standard deviations; omitted second-moment
            # tails are far below the 1e-11 tolerance, even in the broad branch.
            dx = 80.0 / intervals
            moments = [[], [], []]
            for i in range(intervals + 1):
                x = -40.0 + i*dx
                coefficient = 1 if i in (0, intervals) else 4 if i % 2 else 2
                density = normal(x, 10) * (0.5*normal(x, 1) + 0.5*normal(0, 100))
                for k in range(3):
                    moments[k].append(coefficient * density * x**k)
            z, first, second = [math.fsum(values) * dx/3 for values in moments]
            self.assertAlmostEqual(truth["mean"][0], first/z, places=11)
            self.assertAlmostEqual(truth["variance"][0], second/z - (first/z)**2, places=11)
            self.assertAlmostEqual(truth["log_z"], math.log(z), places=11)

    def test_gaussian_hybrid_source_and_sampling_boundary(self):
        source = (HERE / "gaussian-gaussian-hybrid.si").read_text()
        for term in ("let input = 0 in", "bernoulli(0.5)", "gaussian(0,10)",
                     "observe(gaussian(x,1), input)", "observe(gaussian(0,100), input)"):
            self.assertIn(term, source)
        generated = (HERE / "transformed/is-lw-static/gaussian-gaussian-hybrid.txt").read_text()
        self.assertEqual(len(re.findall(r"\bassume\b", generated)), 2)
        self.assertEqual(len(re.findall(r"\bobserve\b", generated)), 2)
        self.assertLess(generated.index("(Gaussian t1 t2)"), generated.index("match b with true"))
        self.assertIsNone(adapt_posterior_readout(generated)[0])

    def test_closed_form_posteriors(self):
        coin = reference("coin")
        self.assertAlmostEqual(coin["mean"][0], 0.8)
        self.assertAlmostEqual(coin["variance"][0], 16 / 1100)
        self.assertAlmostEqual(math.exp(coin["log_z"]), 1 / 72)
        gaussian = reference("gaussian-gaussian")
        self.assertAlmostEqual(gaussian["mean"][0], 30 / 11)
        self.assertAlmostEqual(gaussian["variance"][0], 10 / 11)
        tree = reference("gtree")
        variance = 1 / (1/100 + 1/1010 + 1/1000)
        self.assertAlmostEqual(tree["variance"][0], variance)
        self.assertAlmostEqual(tree["mean"][0], variance * (2.3/1010 - 55/1000))

    def test_independent_batch_conditioning(self):
        for name in ("gaussian-gaussian", "gtree", "kalman", "wheels"):
            with self.subTest(model=name):
                m, p, obs, _, _ = gaussian_problem(name)
                actual_m, actual_p, actual_z = condition(m, p, obs)
                expected_m, expected_v, expected_z = batch_condition(m, p, obs)
                self.assertAlmostEqual(actual_z, expected_z, places=8)
                for i in range(len(m)):
                    self.assertAlmostEqual(actual_m[i], expected_m[i], places=8)
                    self.assertAlmostEqual(actual_p[i][i], expected_v[i], places=8)

    def test_kalman_scalar_recursion(self):
        m, p, z = 0.0, 100.0, 0.0
        for y in range(1, 10):
            p += 1.0
            innovation = p + 1.0
            z -= 0.5 * (math.log(2*math.pi*innovation) + (y-m)**2 / innovation)
            m += p / innovation * (y-m)
            p /= innovation
        actual = reference("kalman")
        self.assertAlmostEqual(actual["mean"][0], m)
        self.assertAlmostEqual(actual["variance"][0], p)
        self.assertAlmostEqual(actual["log_z"], z)

    def test_outlier_against_batch_gaussian_mixture(self):
        # Independent full-joint conditioning, not the scalar reference filter.
        prior, covariance, observations, _, _ = gaussian_problem("kalman")
        components = []
        for mask in product((False, True), repeat=9):
            retained = [obs for obs, keep in zip(observations, mask) if keep]
            means, variances, log_z = batch_condition(prior, covariance, retained)
            for y, keep in zip(OUTLIER_DATA, mask):
                log_z += math.log(0.9 if keep else 0.1)
                if not keep:
                    log_z -= 0.5 * (math.log(200 * math.pi) + y*y/100)
            components.append((math.exp(log_z), means[-1], variances[-1]))
        evidence = math.fsum(w for w, _, _ in components)
        mean = math.fsum(w*m for w, m, _ in components) / evidence
        variance = math.fsum(w*(v+(m-mean)**2) for w, m, v in components) / evidence
        truth = reference("outlier")
        self.assertAlmostEqual(truth["mean"][0], mean, places=11)
        self.assertAlmostEqual(truth["variance"][0], variance, places=11)
        self.assertAlmostEqual(truth["log_z"], math.log(evidence), places=11)
        self.assertEqual(truth["enumerated_configurations"], 512)

    def test_slam_against_position_transition_enumeration(self):
        # Enumerate position paths, marginalizing slips via transition matrices.
        transitions = (((Fraction(1, 10), Fraction(9, 10)), (0, 1)),
                       ((1, 0), (Fraction(9, 10), Fraction(1, 10))))
        rows = []
        for c0, c1, x1, x2 in product((0, 1), repeat=4):
            cells = (c0, c1)
            weight = Fraction(1, 4) * transitions[0][0][x1] * transitions[1][x1][x2]
            for x in (x1, x2):
                weight *= Fraction(1 if cells[x] else 9, 10)
            rows.append((weight, (c0, c1, x1, x2)))
        evidence = sum(w for w, _ in rows)
        means = [sum(w*x[i] for w, x in rows) / evidence for i in range(4)]
        truth = reference("slam")
        self.assertEqual(evidence, Fraction(truth["exact_evidence"]))
        self.assertEqual(means, list(map(Fraction, truth["exact_mean"])))
        for i in range(4):
            for j in range(4):
                covariance = sum(w*(x[i]-means[i])*(x[j]-means[j]) for w, x in rows) / evidence
                self.assertEqual(covariance, Fraction(truth["exact_covariance"][i][j]))

    def test_new_model_data_and_loop_shapes(self):
        for suffix in ("si", "dppl"):
            slam = (HERE / f"slam.{suffix}").read_text()
            data = re.search(r"let data = \[(.*?)\]", slam).group(1)
            self.assertEqual(tuple(tuple(map(int, p)) for p in re.findall(r"\((-?\d+),\s*(-?\d+)\)", data)), SLAM_DATA)
        self.assertIn("List.range(1, 10)", (HERE / "outlier.si").read_text())
        self.assertIn("create 9 (lam i. addi i 1)", (HERE / "outlier.dppl").read_text())
        for name, sites in (("outlier", 2), ("slam", 1)):
            generated = (HERE / f"transformed/is-lw-static/{name}.txt").read_text()
            self.assertIn("foldl", generated)
            self.assertEqual(len(re.findall(r"\bobserve\b", generated)), sites)
            self.assertIsNone(adapt_posterior_readout(generated)[0])

    def test_wheels_smoothing_and_data(self):
        # The first state's posterior uses future observations too.
        m, p, obs, _, _ = gaussian_problem("wheels")
        first_m, _, _ = condition(m, p, obs[:2])
        self.assertGreater(abs(first_m[0] - reference("wheels")["mean"][0]), 0.1)
        for suffix in ("si", "dppl"):
            text = (HERE / f"wheels.{suffix}").read_text()
            pairs = re.findall(r"\[\s*(\d+\.\d+)\s*,\s*(-\d+\.\d+)\s*\]", text)
            self.assertEqual([tuple(map(float, pair)) for pair in pairs], list(WHEELS_DATA))
        text = (HERE / "coin.si").read_text()
        self.assertEqual(text.count(", true)"), 7)
        self.assertEqual(text.count(", false)"), 1)

    def test_reject_invalid_results(self):
        with self.assertRaises(ValueError):
            parse_coreppl("1 1 nan 1\n0\n0\n", 1, 1)
        with self.assertRaises(ValueError):
            parse_coreppl("1 1 0 1\n0 1\n0 1\n", 1, 1)
        truth = reference("coin")
        self.assertEqual(error_metrics(truth["mean"], truth["variance"], truth["log_z"], truth)["mean_rmse"], 0)

    def test_failed_experiments_are_not_accuracy_results(self):
        rows = [{"engine": "coreppl", "model": "kalman", "method": "smc-bpf",
                 "mode": "static", "particles": 10, "status": "runtime_error",
                 "returncode": -11}]
        group = summarize(rows)[0]
        self.assertEqual(group["runs"], 1)
        self.assertEqual(group["successful_runs"], 0)
        self.assertIsNone(group["inference_seconds_median"])
        self.assertIsNone(group["mean_rmse_rms"])

    def test_comparison_excludes_sampling_controls(self):
        native = {"engine": "coreppl", "mode": "static", "requested_readout": "posterior", "readout": "samples"}
        self.assertTrue(comparison_row(native))  # Required draws in unchanged DS.
        self.assertFalse(comparison_row({**native, "mode": "plain"}))
        self.assertFalse(comparison_row({**native, "requested_readout": "samples"}))
        self.assertTrue(comparison_row({"engine": "siren"}))


class CompiledShimTests(unittest.TestCase):
    def test_nonuniform_weights(self):
        directory = HERE / "_build/shim-validation"
        directory.mkdir(parents=True, exist_ok=True)
        source = directory / "check.dppl"
        source.write_text(f'''include "{HERE / 'benchmark-runtime.mc'}"
mexpr
report 0.0 0.0 [[0.0, 2.0], [2.0, 6.0]] [(log 0.25), (log 0.75)]
''')
        command = [str(REPO / "build/cppl"), str(source), "--output", str(directory / "run")]
        built = subprocess.run(command, cwd=directory, env=compiler_environment(), capture_output=True, text=True, timeout=120)
        self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
        output = subprocess.check_output([str(directory / "run")], text=True, cwd=directory)
        result = parse_coreppl(output, 2, 1)[0]
        self.assertEqual(result["mean"], [1.5, 5.0])
        self.assertEqual(result["variance"], [0.75, 3.0])
        self.assertAlmostEqual(result["ess"], 1.6)

    def test_total_variance_for_mixture_components(self):
        directory = HERE / "_build/moment-validation"
        directory.mkdir(parents=True, exist_ok=True)
        source = directory / "check.dppl"
        source.write_text(f'''include "{HERE / 'benchmark-runtime.mc'}"
mexpr
reportMoments 0.0 0.0 [[0.0, 2.0], [2.0, 6.0]] [[4.0, 1.0], [9.0, 4.0]] [(log 0.25), (log 0.75)]
''')
        built = subprocess.run([str(REPO / "build/cppl"), str(source), "--output", str(directory / "run")],
                               cwd=directory, env=compiler_environment(), capture_output=True, text=True, timeout=120)
        self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
        output = subprocess.check_output([str(directory / "run")], text=True, cwd=directory)
        result = parse_coreppl(output, 2, 1)[0]
        self.assertEqual(result["mean"], [1.5, 5.0])
        self.assertEqual(result["variance"], [8.5, 6.25])


class PosteriorReadoutTests(unittest.TestCase):
    def test_compiled_fold_preserves_weights_and_intermediate_draws(self):
        # Instrument the original callback to print the moments of its actual
        # final draw. At N=1, a shared seed must give the same preceding path,
        # likelihood, and conditional moments as the adapted loop.
        original = (HERE / "transformed/is-lw-static/kalman.txt").read_text()
        adapted, _ = adapt_final_scalar_fold(original)
        instrumented = original.replace(
            "        x\n    in\n",
            "        printVector [postMu, (mulf postSigma postSigma)];\n        x\n    in\n")
        self.assertNotEqual(original, instrumented)
        instrumented = re.sub(r"\n([A-Za-z_][A-Za-z_0-9']*)\s*\Z", r"\n(\1, 0.0)\n", instrumented)
        binaries = []
        for name, body in (("original", instrumented), ("adapted", adapted)):
            directory = HERE / "_build/fold-readout-validation" / name
            directory.mkdir(parents=True, exist_ok=True)
            source = directory / "check.dppl"
            source.write_text(wrapper_source("kalman", "is-lw", transformed_body=body))
            binary = directory / "run"
            built = subprocess.run([str(REPO / "build/cppl"), str(source), "--output", str(binary)],
                                   cwd=directory, env=compiler_environment(), capture_output=True, text=True, timeout=120)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            binaries.append(binary)
        from benchmark import parse_float
        for seed in (7, 101, 456):
            raw = [subprocess.check_output([str(binary), "1", str(seed), "1"], cwd=binary.parent, text=True)
                   for binary in binaries]
            lines = raw[0].splitlines()
            self.assertEqual(len(lines), 12)  # Nine loop entries + three report lines.
            expected = list(map(parse_float, lines[8].split()))
            baseline = parse_coreppl("\n".join(lines[9:]), 1, 1)[0]
            actual = parse_coreppl(raw[1], 1, 1)[0]
            self.assertEqual(actual["mean"][0], expected[0])
            self.assertEqual(actual["variance"][0], expected[1])
            self.assertEqual(actual["log_z"], baseline["log_z"])

    def test_scalar_fold_readout_preserves_updates(self):
        original = (HERE / "transformed/is-lw-static/kalman.txt").read_text()
        adapted, audit = adapt_final_scalar_fold(original)
        self.assertIsNotNone(adapted)
        self.assertIn(audit["unchanged_callback_body"], adapted)
        self.assertEqual(original.count("observe"), adapted.count("observe"))
        self.assertEqual(original.count("foldl"), adapted.count("foldl"))
        prefix_length = audit["unchanged_prefix_characters"]
        self.assertEqual(original[:prefix_length], adapted[:prefix_length])
        # Subsequent uses, unsupported families, and name collisions are unsafe.
        for bad in (original + "observe true (Bernoulli 0.5);\n",
                    original.replace("Gaussian postMu postSigma", "Gamma postMu postSigma"),
                    "let readoutPendingState = 0.0 in\n" + original):
            self.assertIsNone(adapt_final_scalar_fold(bad)[0])

    def test_only_terminal_outermost_draws_are_adapted(self):
        rejected = [
            "let x = assume (Gaussian mu sigma) in\nobserve y (Gaussian x noise);\nx\n",
            "lam unit.\n  let x = assume (Gaussian mu sigma) in\n  x\n",
            "let x = assume (Gamma alpha beta) in\nx\n",
            "let x = assume (Gaussian mu sigma) in\naddf x x\n",
            "let x = assume (Gaussian (f z) sigma) in\nx\n",
        ]
        for text in rejected:
            with self.subTest(text=text):
                adapted, audit = adapt_terminal_draw(text)
                self.assertIsNone(adapted)
                self.assertFalse(audit["applied"])
        prefix = "let internal = assume (Gaussian m s) in\nobserve y (Gaussian internal r);\n"
        text = prefix + "let result = assume (Beta alphaNew betaNew) in\nresult\n"
        adapted, audit = adapt_terminal_draw(text)
        self.assertEqual(adapted[:len(prefix)], prefix)
        self.assertEqual(audit["remaining_assume_sites"], 1)
        self.assertIn("expectation (Beta alphaNew betaNew)", adapted)

    def test_existing_compiler_outputs_and_bias_are_preserved(self):
        for name, remaining in (("coin", 0), ("gaussian-gaussian", 0), ("gtree", 1)):
            original = (HERE / f"transformed/is-lw-static/{name}.txt").read_text()
            adapted, audit = adapt_terminal_draw(original)
            self.assertIsNotNone(adapted)
            prefix_length = audit["unchanged_prefix_characters"]
            self.assertEqual(adapted[:prefix_length], original[:prefix_length])
            self.assertEqual(audit["remaining_assume_sites"], remaining)
        for name in ("kalman", "wheels"):
            original = (HERE / f"transformed/is-lw-static/{name}.txt").read_text()
            self.assertIsNone(adapt_terminal_draw(original)[0])

    def test_identical_posterior_in_every_particle(self):
        # Check *every* returned pair inside CorePPL, not just its aggregate.
        for name in ("coin", "gaussian-gaussian"):
            with self.subTest(model=name):
                original = (HERE / f"transformed/is-lw-static/{name}.txt").read_text()
                adapted, _ = adapt_terminal_draw(original)
                wrapper = wrapper_source(name, "is-lw", transformed_body=adapted)
                check = '''let result = distEmpiricalSamples dist in
  let first = head result.0 in
  iter (lam x. if eqf x.0 first.0 then
    (if eqf x.1 first.1 then () else exit 7) else exit 8) result.0;
  '''
                wrapper = wrapper.replace("let result = distEmpiricalSamples dist in", check)
                directory = HERE / "_build/identical-particle-validation" / name
                directory.mkdir(parents=True, exist_ok=True)
                source = directory / "check.dppl"
                source.write_text(wrapper)
                built = subprocess.run([str(REPO / "build/cppl"), str(source), "--output", str(directory / "run")],
                                       cwd=directory, env=compiler_environment(), capture_output=True, text=True, timeout=120)
                self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
                means, variances = [], []
                for n in (1, 17, 10000):
                    for seed in (123, 456):
                        output = subprocess.check_output([str(directory / "run"), str(n), str(seed), "1"],
                                                         cwd=directory, text=True)
                        result = parse_coreppl(output, 1, 1)[0]
                        means.append(result["mean"][0])
                        variances.append(result["variance"][0])
                self.assertEqual(len(set(means)), 1)
                self.assertEqual(len(set(variances)), 1)
                self.assertAlmostEqual(means[0], reference(name)["mean"][0], places=15)
                self.assertAlmostEqual(variances[0], reference(name)["variance"][0], places=14)


if __name__ == "__main__":
    unittest.main()
