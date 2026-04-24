Ниже версия для results.md, без лишних разговорных фраз.

# LunarLander GRPO Reward Suite Results
## Context
Это уже не random-init suite, а suite с `competent_anchor`: политика стартует не с нуля, а с уже компетентного checkpoint `lunarlander_baseline_clean_seed42.pt`.
Поэтому интерпретация такая: эксперимент проверяет не то, “может ли GRPO научиться с нуля”, а то, **улучшает или портит конкретная reward-схема уже компетентную политику**.
Все runs использовали:
- environment: `LunarLander-v3`
- initialization: `competent_anchor`
- fixed validation set: 32 seeds
- deterministic eval: `1.0`
- full eval coverage: `32/32`
## Compared reward variants
| Run | Reward scheme |
|---|---|
| `terminal_only` | только terminal success reward |
| `terminal_smooth` | terminal + smoothness |
| `terminal_sub_prog` | terminal + subgoal + progress |
| `full` | terminal + subgoal + progress + smoothness |
| `dense_only` | subgoal + progress + smoothness, без terminal reward |
Общая reward-формула:
\[
R = w_{sub}r_{sub} + w_{prog}(r_{prog}+r_{micro}) + w_{smooth}r_{smooth} + w_{final}r_{final}
\]
---
# 1. Success metrics
## Final Train Success
`Final Train Success` — это доля успешных train-эпизодов в последнем logged train-window.
| Run | Final Train Success |
|---|---:|
| `terminal_only` | `0.7422` |
| `terminal_smooth` | `0.4922` |
| `terminal_sub_prog` | `0.9609` |
| `full` | `0.5547` |
| `dense_only` | `0.0234` |
Главный результат: `terminal_sub_prog` дал лучший final train success: `0.9609`.
`dense_only` почти полностью деградировал к финалу: final train success всего `0.0234`.
## Best Train Success
`Best Train Success` — это лучший train success, достигнутый в любой момент run-а, не обязательно в конце.
| Run | Best Train Success |
|---|---:|
| `terminal_only` | `0.9609` |
| `terminal_smooth` | `0.9609` |
| `terminal_sub_prog` | `0.9766` |
| `full` | `0.9844` |
| `dense_only` | `0.9609` |
Почти все runs в какой-то момент достигали хорошего train-поведения. Но часть reward-схем затем деградировала к финалу.
Самый показательный случай — `dense_only`: best train success `0.9609`, но final train success `0.0234`.
Это означает, что dense-only reward может увести уже компетентную политику от успешного поведения.
## Final Val Success
`Final Val Success` — основная метрика итогового качества политики: доля успешных эпизодов на фиксированных 32 validation seeds в финальном eval.
| Run | Final Val Success |
|---|---:|
| `terminal_only` | `1.0000` |
| `terminal_smooth` | `0.4375` |
| `terminal_sub_prog` | `1.0000` |
| `full` | `0.7812` |
| `dense_only` | `0.0938` |
`terminal_only` и `terminal_sub_prog` идеально проходят финальную validation-выборку.
`full` заметно хуже: `0.7812`.
`terminal_smooth` сильно хуже: `0.4375`.
`dense_only` почти ломает поведение: `0.0938`.
## Best Val Success
`Best Val Success` — лучший validation success за весь run.
| Run | Best Val Success |
|---|---:|
| `terminal_only` | `1.0000` |
| `terminal_smooth` | `0.9688` |
| `terminal_sub_prog` | `1.0000` |
| `full` | `0.9688` |
| `dense_only` | `0.9688` |
Даже runs с плохим final validation success в какой-то момент почти идеально проходили validation.
Особенно важно для `dense_only`: best val success `0.9688`, но final val success `0.0938`.
Это показывает, что dense-only objective не просто “не помогает”, а может разрушить уже компетентную политику в процессе GRPO-обновлений.
---
# 2. Reward metrics
## Final Train Reward
`Final Train Reward` — средняя итоговая train-награда в последнем logged train-window.
| Run | Final Train Reward |
|---|---:|
| `terminal_only` | `0.7422` |
| `terminal_smooth` | `0.3324` |
| `terminal_sub_prog` | `1.1987` |
| `full` | `0.4542` |
| `dense_only` | `-0.0633` |
Для `terminal_only` reward совпадает с success, потому что reward состоит только из terminal success.
Для shaped runs reward может быть выше success, потому что добавляются `subgoal` и `progress`.
`terminal_sub_prog` даёт высокий reward `1.1987` и одновременно высокий train success `0.9609`. Это значит, что shaped-компоненты добавляют полезный сигнал и не ломают задачу.
`dense_only` имеет отрицательный train reward `-0.0633` и почти нулевой final train success `0.0234`.
## Final Val Reward
`Final Val Reward` — то же самое, но на validation.
| Run | Final Val Reward |
|---|---:|
| `terminal_only` | `1.0000` |
| `terminal_smooth` | `0.2934` |
| `terminal_sub_prog` | `1.2347` |
| `full` | `0.7966` |
| `dense_only` | `0.0146` |
`terminal_sub_prog` имеет val reward выше, чем `terminal_only`, потому что помимо успешной посадки получает дополнительный `subgoal/progress` reward.
При этом validation success у `terminal_sub_prog` не выше, чем у `terminal_only`, потому что оба уже достигают `1.0000`.
---
# 3. Delta vs terminal_only
На графиках `vs terminal_only` показана разница:
\[
\Delta = metric(run) - metric(terminal\_only)
\]
Если значение выше нуля, run лучше `terminal_only` по этой метрике. Если ниже нуля, хуже.
## Final Train Success delta
| Run | Delta vs `terminal_only` |
|---|---:|
| `terminal_smooth` | `-0.2500` |
| `terminal_sub_prog` | `+0.2188` |
| `full` | `-0.1875` |
| `dense_only` | `-0.7188` |
Только `terminal_sub_prog` улучшил final train success относительно `terminal_only`.
`smoothness`, `full` и `dense_only` ухудшили финальное train-поведение.
## Final Val Success delta
| Run | Delta vs `terminal_only` |
|---|---:|
| `terminal_smooth` | `-0.5625` |
| `terminal_sub_prog` | `0.0000` |
| `full` | `-0.2188` |
| `dense_only` | `-0.9062` |
Ни один run не превзошёл `terminal_only` по final validation success, потому что `terminal_only` уже достиг `1.0000`.
`terminal_sub_prog` сохранил идеальный validation success.
Все остальные shaped-варианты ухудшили validation.
---
# 4. Fuel proxy
`Fuel Proxy` — proxy на активность двигателей / расход топлива. Большее значение обычно означает более активное управление.
## Train fuel
| Run | Final Train Fuel Proxy |
|---|---:|
| `terminal_only` | `209.52` |
| `terminal_smooth` | `133.33` |
| `terminal_sub_prog` | `231.70` |
| `full` | `148.45` |
| `dense_only` | `101.88` |
## Val fuel
| Run | Final Val Fuel Proxy |
|---|---:|
| `terminal_only` | `246.47` |
| `terminal_smooth` | `138.50` |
| `terminal_sub_prog` | `240.41` |
| `full` | `147.72` |
| `dense_only` | `107.06` |
`terminal_smooth`, `full` и `dense_only` сильно снижают fuel proxy.
Но это не означает улучшение: у этих runs одновременно падает success.
Интерпретация: smoothness/dense reward заставляют политику меньше управлять, но это ухудшает посадку. В этой постановке низкий fuel proxy не является самостоятельным показателем качества.
`terminal_sub_prog` почти не снижает fuel относительно `terminal_only`, но даёт лучший train success и сохраняет идеальный validation success.
---
# 5. Phase transitions
`Phase Transitions` — среднее число переходов между фазами, например:
- `ALIGN -> DESCEND`
- `DESCEND -> TOUCHDOWN`
Для validation:
| Run | Final Val Phase Transitions |
|---|---:|
| `terminal_only` | `2.0000` |
| `terminal_smooth` | `1.5938` |
| `terminal_sub_prog` | `2.0000` |
| `full` | `1.7188` |
| `dense_only` | `1.5312` |
`terminal_only` и `terminal_sub_prog` проходят почти полный фазовый путь.
`full`, `terminal_smooth` и `dense_only` имеют меньше phase transitions, что согласуется с их худшим success.
Фазы здесь используются только как диагностическая метрика: меньше переходов обычно означает, что политика хуже доходит до поздних стадий посадки.
---
# 6. Phase counts and APPROACH
`APPROACH` не был использован ни в одном run.
`micro-progress` также не сработал ни в одном run.
Это значит:
- текущий phase classifier фактически работает через `ALIGN`, `DESCEND`, `TOUCHDOWN`;
- `APPROACH` как отдельная стадия в этом suite не участвовал;
- `r_micro`, который зависит от `APPROACH`, был полностью неактивен.
Поэтому по этим данным нельзя делать вывод, полезен или вреден `micro-progress`: он просто не был протестирован как активная reward-компонента.
При этом важно, что успешная схема `terminal_sub_prog` работает без `APPROACH` и без `micro-progress`.
---
# 7. Reward composition
Reward composition показывает, из каких компонент состоит итоговая награда.
## terminal_only
Reward состоит только из final success:
| Split | Final share |
|---|---:|
| Train | `0.7422` |
| Val | `1.0000` |
Вся reward-логика определяется успешной посадкой.
## terminal_sub_prog
Train reward shares:
| Component | Share |
|---|---:|
| subgoal | `0.0812` |
| progress | `0.1422` |
| final | `0.7765` |
Val reward shares:
| Component | Share |
|---|---:|
| subgoal | `0.0681` |
| progress | `0.1216` |
| final | `0.8103` |
Это наиболее сбалансированная структура:
- final reward остаётся доминирующим;
- subgoal/progress дают дополнительный shaping;
- dense-компоненты не вытесняют terminal success.
## full
Train reward shares:
| Component | Share |
|---|---:|
| subgoal | `0.0899` |
| progress | `0.2099` |
| smoothness | `0.2923` |
| final | `0.4079` |
Val reward shares:
| Component | Share |
|---|---:|
| subgoal | `0.0860` |
| progress | `0.1622` |
| smoothness | `0.1658` |
| final | `0.5860` |
В `full` smoothness занимает слишком большую долю reward, особенно на train: около `29%`.
Final share падает до `40.8%` на train.
Это объясняет деградацию: smoothness начинает конкурировать с terminal success и меняет поведение в сторону меньшего управления, а не лучшей посадки.
## dense_only
Train reward shares:
| Component | Share |
|---|---:|
| subgoal | `0.2787` |
| progress | `0.4222` |
| smoothness | `0.2991` |
| final | `0.0000` |
Val reward shares:
| Component | Share |
|---|---:|
| subgoal | `0.2100` |
| progress | `0.5683` |
| smoothness | `0.2218` |
| final | `0.0000` |
`dense_only` не содержит terminal reward.
Итоговый success почти нулевой:
- final train success: `0.0234`
- final val success: `0.0938`
Это показывает, что dense shaping сам по себе не удерживает настоящую цель задачи.
---
# 8. Per-run interpretation
## terminal_only
Baseline control.
Results:
- final train success: `0.7422`
- final val success: `1.0000`
- final val reward: `1.0000`
Так как старт идёт с `competent_anchor`, terminal-only GRPO не портит validation и сохраняет сильную политику.
## terminal_smooth
Results:
- final train success: `0.4922`
- final val success: `0.4375`
- fuel proxy сильно ниже
- phase transitions ниже
Interpretation: smoothness penalty ухудшает посадку. Он снижает активность управления, но вместе с этим снижает success.
## terminal_sub_prog
Лучший shaped-вариант.
Results:
- final train success: `0.9609`
- final val success: `1.0000`
- final train reward: `1.1987`
- final val reward: `1.2347`
- phase transitions почти максимальные
- no warning
Interpretation: subgoal + progress помогают на train и не вредят validation.
## full
Results:
- final train success: `0.5547`
- final val success: `0.7812`
- хуже, чем `terminal_sub_prog`
- smoothness занимает значимую долю reward
Interpretation: добавление smoothness к хорошей схеме `terminal + subgoal + progress` портит результат.
## dense_only
Results:
- final train success: `0.0234`
- final val success: `0.0938`
- terminal reward отсутствует
Interpretation: dense reward без terminal success не удерживает настоящую цель задачи и приводит к деградации политики.
---
# Main conclusion
Лучший reward-вариант для LunarLander в этом suite:
\[
R = 0.1r_{sub} + 0.3r_{prog} + 1.0r_{final}
\]
То есть:
```yaml
weights:
  sub: 0.1
  prog: 0.3
  smooth: 0.0
  final: 1.0

Итоговые выводы:

1. terminal reward обязателен.
2. dense_only почти разрушает итоговое поведение.
3. subgoal + progress полезны как дополнительный shaping.
4. terminal_sub_prog улучшает train success и reward, не ухудшая validation.
5. smoothness в этой постановке вреден: снижает fuel proxy, но ухудшает success.
6. full reward хуже, чем более простая схема terminal + subgoal + progress.
7. APPROACH и micro-progress в этом suite не участвовали, поэтому их полезность здесь не проверена.
8. LunarLander остаётся sanity check для reward plumbing, а не доказательством переносимости reward-схемы на VLA.

На LunarLander распределённая награда работает лучше всего, когда dense shaping ограничен subgoal + progress, а terminal success остаётся доминирующей целью; smoothness и dense-only ухудшают реальную task performance, несмотря на улучшение некоторых proxy-метрик вроде fuel.