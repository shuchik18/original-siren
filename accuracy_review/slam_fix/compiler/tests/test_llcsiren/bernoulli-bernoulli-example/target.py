let rv1 = Prob.new_var() in
let () = Prob.set_distr(rv1, bernoulli(0.5)) in
let () = Prob.set_lookup("z", rv1) in
let z = rv1 in
let rv3 = Prob.new_var() in
let () = Prob.set_distr(rv3, bernoulli(if z then 0.8 else 0.2)) in
let () = Prob.set_distr(Prob.get_par(rv3, 0), bernoulli(((Prob.get_distr(Prob.get_par(rv3, 0))) * Prob.subst_rv(if rv3 then (Prob.get_distr(rv3)) else 1.0 + (-1.0 * (Prob.get_distr(rv3))), Prob.get_par(rv3, 0), true))/(if rv3 then (Prob.get_par(rv3, 0) * Prob.subst_rv((Prob.get_distr(rv3)), Prob.get_par(rv3, 0), true)) + ((1.0 + (-1.0 * (Prob.get_distr(Prob.get_par(rv3, 0))))) * Prob.subst_rv((Prob.get_distr(rv3)), Prob.get_par(rv3, 0), false)) else 1.0 + (-1.0 * ((Prob.get_distr(Prob.get_par(rv3, 0))) * Prob.subst_rv((Prob.get_distr(rv3)), Prob.get_par(rv3, 0), true)) + ((1.0 + (-1.0 * (Prob.get_distr(Prob.get_par(rv3, 0))))) * Prob.subst_rv((Prob.get_distr(rv3)), Prob.get_par(rv3, 0), false)))))) in
#if rv3 then 0.8 else 0.2 where rv3 is parent and if it is true then 0.8 else 0.2
let () = Prob.set_distr(rv3, ((Prob.get_distr(Prob.get_par(rv3, 0))) * Prob.subst_rv((Prob.get_distr(rv3)), Prob.get_par(rv3, 0), true)) + ((1.0 + (-1.0 * (Prob.get_distr(Prob.get_par(rv3, 0))))) * Prob.subst_rv((Prob.get_distr(rv3)), Prob.get_par(rv3, 0), false))) in
# otherwise it is bernoulli(0.5)
let () = Prob.observe_inner(rv3, true) in
let () = Prob.null() in
Prob.finalize(z)