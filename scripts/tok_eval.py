"""
Evaluate compression ratio of the tokenizer.
"""

from nanochat.tokenizer import get_tokenizer, RustBPETokenizer
from nanochat.dataset import parquets_iter_batched

# Random text I got from a random website this morning
news_text = r"""
(Washington, D.C., 9 de julho de 2025) – Ontem, o Serviço Nacional de Sanidade, Inocuidade e Qualidade Agroalimentar do México (SENASICA) relatou um novo caso de miíase causada pelo berne-do-novo-mundo (NWS, na sigla em inglês) em Ixhuatlán de Madero, Veracruz, no México, que fica aproximadamente 257 quilômetros ao norte da atual área de dispersão da mosca estéril, no lado leste do país e a 595 quilômetros ao sul da fronteira entre os EUA e o México. Essa nova detecção ao norte ocorre cerca de dois meses após registros anteriores em Oaxaca e Veracruz, a menos de 1.126 quilômetros da fronteira dos EUA, o que levou ao fechamento de nossos portos para bovinos, bisões e cavalos mexicanos em 11 de maio de 2025.

Embora o USDA tenha anunciado uma estratégia de reabertura gradual e baseada em risco dos portos para bovinos, bisões e equídeos provenientes do México a partir de 7 de julho de 2025, este novo caso de berne-do-novo-mundo relatado levanta preocupações significativas sobre as informações previamente compartilhadas por autoridades mexicanas e compromete seriamente o cronograma de reabertura de cinco portos previsto para o período de 7 de julho a 15 de setembro. Portanto, para proteger o rebanho americano e o abastecimento alimentar do nosso país, a Secretária Rollins ordenou o fechamento imediato do comércio de animais vivos através dos portos de entrada ao sul.

"Os Estados Unidos prometeram ser vigilantes — e, após a detecção deste novo caso de berne-do-novo-mundo, estamos pausando as reaberturas planejadas dos portos para intensificar a quarentena e combater esta praga mortal no México. Precisamos ver avanços adicionais no combate ao berne-do-novo-mundo em Veracruz e outros estados mexicanos próximos para reabrir os portos de entrada de animais vivos na fronteira sul," disse a Secretária de Agricultura dos EUA, Brooke L. Rollins. "Graças à rigorosa vigilância das equipes do USDA nos EUA e no México, conseguimos agir de forma rápida e decisiva para responder à disseminação desta praga mortal.""".strip()

# Random Korean text (to test non-English compression)
korean_text = r"""
정직한 사실 위에, 공정한 시선을 더하다
Herald Korea Times

헤럴드코리아타임즈는 정치, 경제, 사회, 문화 등 한국 사회 전반의 주요 이슈를 심도 있게 다루는 종합 온라인 신문사입니다.

우리는 단순히 뉴스를 전달하는 것이 아니라, 사실(Fact)에 기반한 양측의 시각을 균형 있게 조명하며, 독자 여러분이 스스로 판단할 수 있는 ‘정보의 균형’을 제공합니다.

한국 언론의 오랜 문제로 지적되어 온 정치적 편향, 이념적 왜곡에서 벗어나
오직 정직함과 공정함을 원칙으로 삼는 언론을 지향합니다.
어느 한쪽의 주장만을 확대하거나 감추지 않고,
**모든 쟁점에 대해 ‘무엇이 쟁점인지’, ‘누가 무엇을 주장하는지’, ‘사실은 무엇인지’**를 명확히 전달하는 데 집중합니다.
""".strip()

# Random piece of code
code_text = r"""
class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = merge(ids, pair, idx)
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
""".strip()

math_text = r"""
\documentclass[12pt]{article}
\usepackage{amsmath,amsthm,amssymb}
\usepackage[margin=1in]{geometry}

\newtheorem{theorem}{Theorem}
\newtheorem*{remark}{Remark}

\begin{document}

\begin{center}
{\Large Uma Identidade Fofa: A Soma de Cubos é um Quadrado}
\end{center}

\begin{theorem}
Para todo número inteiro $n \ge 1$,
\[
\sum_{k=1}^{n} k^{3} \;=\; \left(\frac{n(n+1)}{2}\right)^{2}.
\]
\end{theorem}

\begin{proof}[Prova 1 (Indução)]
Deixe $S(n) = \sum_{k=1}^{n} k^3$. Para $n=1$, $S(1)=1=(1\cdot 2/2)^2$, então o caso base é válido.

Assuma que $S(n)=\big(\tfrac{n(n+1)}{2}\big)^2$ para algum $n\ge 1$.
Então
\[
S(n+1)
= S(n) + (n+1)^3
= \left(\frac{n(n+1)}{2}\right)^2 + (n+1)^3.
\]
Fatorando $(n+1)^2$:
\[
S(n+1)
= (n+1)^2\left( \frac{n^2}{4} + (n+1) \right)
= (n+1)^2\left( \frac{n^2 + 4n + 4}{4} \right)
= (n+1)^2\left( \frac{(n+2)^2}{4} \right).
\]
Assim
\[
S(n+1)=\left(\frac{(n+1)(n+2)}{2}\right)^2,
\]
que corresponde à fórmula afirmada com $n$ substituído por $n+1$. Por indução, a identidade vale para todo $n\ge 1$.
\end{proof}

\begin{proof}[Prova 2 (Telescoping algébrico)]
Recorde a identidade binomial
\[
(k+1)^4 - k^4 = 4k^3 + 6k^2 + 4k + 1.
\]
Somando ambos os lados de $k=0$ a $n$ telescopa:
\[\
(n+1)^4 - 0^4
= \sum_{k=0}^{n}\big(4k^3 + 6k^2 + 4k + 1\big)
= 4\sum_{k=1}^{n}k^3 + 6\sum_{k=1}^{n}k^2 + 4\sum_{k=1}^{n}k + (n+1).
\]
Usando as somas padrão
\[
\sum_{k=1}^{n}k = \frac{n(n+1)}{2}
\quad\text{and}\quad
\sum_{k=1}^{n}k^2 = \frac{n(n+1)(2n+1)}{6},
\]
resolvendo para $\sum_{k=1}^{n}k^3$ obtemos
\[
\sum_{k=1}^{n}k^3 = \left(\frac{n(n+1)}{2}\right)^2.
\]
\end{proof}

\begin{remark}
Geometricamente, a identidade diz: ``somar $1^3,2^3,\dots,n^3$ constrói um quadrado perfeito’’—nomeadamente o quadrado do $n$-ésimo número triangular. É por isso que às vezes se chama ao fenômeno \emph{soma dos cubos é um quadrado}.
\end{remark}

\end{document}
""".strip()

science_text = r"""
A fotossíntese é um processo de transdução de energia fotquímica no qual complexos pigmento-proteína captadores de luz, localizados nas membranas dos tilacoides de fototróficos oxigênicos, absorvem fótons e iniciam a separação de cargas no centro de reação, impulsionando a cadeia de transporte eletrônico linear da água ao NADP⁺ via fotossistema II, o complexo citocromo b₆f e o fotossistema I, gerando concomitantemente uma força motriz próton trans-tilacoide utilizada pela ATP sintase cloroplástica. As reações dependentes da luz produzem ATP e NADPH, que alimentam o ciclo Calvin–Benson–Bassham no estroma, onde a ribulose-1,5-bisfosfato é carboxilada pela ribulose-1,5-bisfosfato carboxilase/oxigenase (RuBisCO) para formar 3-fosfoglicerato, subsequentemente reduzido e regenerado através de uma série de etapas enzimáticas, permitindo a assimilação líquida de CO₂ em trioses fosfatos e, finalmente, carboidratos. Este processo é rigidamente regulado por mecanismos fotoprotetores, feedback redox e fluxo de metabólitos, representando uma via bioquímica central que acopla a captura de energia solar à produtividade primária da biosfera.
""".strip()

# The tokenizer was trained on data from earlier shards, so it has seen this data
train_docs = next(parquets_iter_batched(split="train"))
train_text = "\n".join(train_docs)
val_docs = next(parquets_iter_batched(split="val"))
val_text = "\n".join(val_docs)

all_text = [
    ("news", news_text),
    ("korean", korean_text),
    ("code", code_text),
    ("math", math_text),
    ("science", science_text),
    ("fwe-train", train_text),
]
if val_text:
    all_text.append(("fwe-val", val_text))

# Try out current default compared to GPT-2 and GPT-4 tokenizers
tokenizer_results = {}
vocab_sizes = {}

for tokenizer_name in ["gpt2", "gpt4", "ours"]:

    if tokenizer_name == "gpt2":
        tokenizer = RustBPETokenizer.from_pretrained("gpt2") # gpt-2 base model tokenizer
    elif tokenizer_name == "gpt4":
        tokenizer = RustBPETokenizer.from_pretrained("cl100k_base") # gpt-4 base model tokenizer
    else:
        tokenizer = get_tokenizer()

    vocab_sizes[tokenizer_name] = tokenizer.get_vocab_size()
    tokenizer_results[tokenizer_name] = {}

    for name, text in all_text:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

        encoded_bytes = text.encode('utf-8')
        ratio = len(encoded_bytes) / len(encoded)
        tokenizer_results[tokenizer_name][name] = {
            'bytes': len(encoded_bytes),
            'tokens': len(encoded),
            'ratio': ratio
        }

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

# Print vocab sizes
print(f"\nVocab sizes:")
print(f"GPT-2: {vocab_sizes['gpt2']}")
print(f"GPT-4: {vocab_sizes['gpt4']}")
print(f"Ours: {vocab_sizes['ours']}")

def print_comparison(baseline_name, baseline_results, ours_results, all_text):
    """Print comparison table between baseline tokenizer and ours."""
    print(f"\nComparison with {baseline_name}:")
    print("=" * 95)
    print(f"{'Text Type':<10} {'Bytes':<8} {baseline_name:<15} {'Ours':<15} {'Relative':<12} {'Better':<10}")
    print(f"{'':10} {'':8} {'Tokens':<7} {'Ratio':<7} {'Tokens':<7} {'Ratio':<7} {'Diff %':<12}")
    print("-" * 95)

    for name, text in all_text:
        baseline_data = baseline_results[name]
        ours_data = ours_results[name]

        # Calculate relative difference (positive means ours is better, negative means worse)
        # Using tokens: fewer tokens is better, so we calculate (baseline_tokens - ours_tokens) / baseline_tokens
        relative_diff = ((baseline_data['tokens'] - ours_data['tokens']) / baseline_data['tokens']) * 100

        # Determine which has better compression (higher ratio = better)
        if baseline_data['ratio'] > ours_data['ratio']:
            baseline_color, ours_color = GREEN, RED
            better = baseline_name
            diff_color = RED
        elif ours_data['ratio'] > baseline_data['ratio']:
            baseline_color, ours_color = RED, GREEN
            better = "Ours"
            diff_color = GREEN
        else:
            baseline_color, ours_color = "", ""
            better = "Tie"
            diff_color = ""

        print(f"{name:<10} {baseline_data['bytes']:<8} "
              f"{baseline_color}{baseline_data['tokens']:<7}{RESET} "
              f"{baseline_color}{baseline_data['ratio']:<7.2f}{RESET} "
              f"{ours_color}{ours_data['tokens']:<7}{RESET} "
              f"{ours_color}{ours_data['ratio']:<7.2f}{RESET} "
              f"{diff_color}{relative_diff:+7.1f}%{RESET}     "
              f"{better:<10}")

# Print comparisons
print_comparison("GPT-2", tokenizer_results['gpt2'], tokenizer_results['ours'], all_text)
print_comparison("GPT-4", tokenizer_results['gpt4'], tokenizer_results['ours'], all_text)

# Log to report
from nanochat.report import get_report
lines = []
for baseline_name in ["GPT-2", "GPT-4"]:
    baseline_key = baseline_name.lower().replace('-', '')
    baseline_results = tokenizer_results[baseline_key]
    ours_results = tokenizer_results['ours']
    lines.append(f"### Comparison with {baseline_name}")
    lines.append("")
    lines.append("| Text Type | Bytes | " + baseline_name + " Tokens | " + baseline_name + " Ratio | Ours Tokens | Ours Ratio | Relative Diff % |")
    lines.append("|-----------|-------|--------------|--------------|-------------|------------|-----------------|")
    for name, text in all_text:
        baseline_data = baseline_results[name]
        ours_data = ours_results[name]
        relative_diff = ((baseline_data['tokens'] - ours_data['tokens']) / baseline_data['tokens']) * 100
        lines.append(f"| {name} | {baseline_data['bytes']} | {baseline_data['tokens']} | {baseline_data['ratio']:.2f} | {ours_data['tokens']} | {ours_data['ratio']:.2f} | {relative_diff:+.1f}% |")
    lines.append("")
report_markdown = "\n".join(lines)
get_report().log(section="Tokenizer evaluation", data=[
    report_markdown,
])
