# Conditional PPO auxiliary target definition

Status: protocol frozen for source implementation; real PPO NOT_RUN. Coordinator decision following Luna's request to resolve an undefined auxiliary target. No current collection change.

G-WM uses a frozen, eval-mode world model to distil one-step public dynamic predictions into a critic auxiliary head. This is model-prediction distillation, not ground-truth dynamic supervision and not an imagined actor rollout.

For the actual executed legal candidate a at state s, the frozen model predicts y(s,a) for progress, energy delta, command delay and risk. The critic predicts z(s,a). Use mean squared error over explicitly valid dimensions, with coefficient 0.1 on the same normalised dynamic scales used for WM training. The validity mask requires a known executed action, its legal-candidate identity, and actual supervision support for that dynamic dimension recorded by the WM checkpoint. Unknown action information or an unsupported dimension is masked; an all-invalid sample does not manufacture an auxiliary update.

Do not distil unexecuted candidate predictions as if they were supervised outcomes. Candidate action encoding and field order must match the frozen WM training encoder. No gain labels enter this loss.

The auxiliary and value heads share the critic encoder. Actor and critic have disjoint parameters. WM receives no gradient and remains unchanged; the actor receives only real-environment PPO gradients. Value/GAE targets use actual Isaac rewards. The G baseline retains the same critic capacity with auxiliary coefficient zero; actor initialisation, scene order and real sampling budget are paired.

Necessary synthetic CPU checks may provide known actions and valid dynamic dimensions to test the gradient path. They do not establish action support in current formal data, real GPU training, or method efficacy. A missing action-support mask in real data cannot be replaced by all ones merely to obtain nonzero auxiliary gradients.
