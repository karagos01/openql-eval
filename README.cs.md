# openQL / openOL — nezávislé vyhodnocení

Reprodukovatelné vyhodnocení práce M. Mazgal, *openQL / openOL: Tensor Matrix
Architecture for Unified Simulation of Physical Fields in 12D and 112D via GPU
Tensor Cores* (Zenodo, 3. října 2026, DOI
[10.5281/zenodo.23112560](https://doi.org/10.5281/zenodo.23112560)).

Text je v [`PAPER.cs.md`](PAPER.cs.md), anglicky v [`PAPER.md`](PAPER.md).

**Kvaternionová polovina je správná a celá** — matice 4×4 v §2.1 je levá regulární
reprezentace ℍ a splňuje M(q)M(r) = M(qr) na 1,8·10⁻¹⁵, a je to první centrální
algebraický krok v preprintech tohoto autora, který obstojí od začátku do konce.
**Oktonionová polovina téže věty je vyloučená teorémem:** násobení matic je
asociativní, takže každý homomorfismus 𝕆 → Mₙ(ℝ) zabije každý asociátor, a
asociátory autorovy vlastní násobicí tabulky generují celou imaginární část, takže
zůstane 1 z 8 dimenzí. Jediná oktonionová citace práce, Baezův přehled, uvádí
nonasociativitu na své první stránce. Dál: uzel 12×12 je blokově diagonální, a tedy
jsou to tři rozvázané kvaterniony; navržený předpis σ(W·S + b) není symplektický; a
bloková maticová reprezentace leží na roofline ve stejném bodě jako ten 112složkový
stav, který kóduje, a stojí přitom 112× více provozu.

> Komentáře v kódu a výpisy skriptů jsou anglicky. Anglická verze textu je
> v [`README.md`](README.md) a [`PAPER.md`](PAPER.md).

## Co je potřeba

```
python3 -m pip install -r requirements.txt   # jen numpy
```

Bez GPU, bez sítě, bez stahování modelů. §§5–6 v `PAPER.cs.md` jsou aritmetika nad
publikovanými hardwarovými specifikacemi, protože preprint žádný kód nepřikládá.

## Reprodukce výsledků

```
./run_all.sh          # všechno, asi 20 sekund na jakémkoli CPU
```

| Skript | Co spočítá | Čas |
|---|---|---|
| `quaternion.py` | matice z §2.1 jako reprezentace ℍ; co vznikne z „concatenating three such state matrices"; dvě odvození dvanáctky | 2 s |
| `octonion.py` | asociátory publikované tabulky, obal, který vyplní, a naměřené selhání téže konstrukce pro 𝕆 | 8 s |
| `symplectic.py` | energie pod σ(W·S + b) s izolovaným σ a b; kodimenze Sp; W nafitovaná na přesná data a rozjetá | 8 s |
| `roofline.py` | režie reprezentace, aritmetická intenzita, rezidentní uzly, kroková frekvence, využití Tensor Cores, atomy na kartu | 1 s |

`fano.py` je násobicí tabulka z doprovodného repozitáře `causal-trilogy-eval`,
nezměněná, takže `octonion.py` měří autorovy vlastní strukturní konstanty. Celý
výstup jednoho běhu je v [`results.log`](results.log).

## Klíčová čísla

| co | publikováno | naměřeno / spočítáno |
|---|---|---|
| matice 4×4 z §2.1 je reprezentace ℍ | „isomorphic" | **správně**, chyba 1,78·10⁻¹⁵ |
| bázové trojice s nenulovým asociátorem | — | 168 z 512, generují všech **7** dimenzí Im 𝕆 |
| dimenze 𝕆, které přežijí jakoukoli maticovou formu | „unified block matrices" | **1 z 8 (12,5 %)**, takže φ je nulové zobrazení |
| tatáž konstrukce, L_x L_y proti L_{xy} | — | ℍ: 1,8·10⁻¹⁵ — **𝕆: 29,5 (relativně 2,76)** |
| uzel 12×12, odezva mimo perturbovaný blok | „unified state structure" | **0,000** — algebra je ℍ⊕ℍ⊕ℍ |
| uzel 12×12, stav proti slotům | — | 12 čísel ve 144 slotech, 96 strukturně nulových |
| odkud je 12 | — | odvozeno dvakrát, nekompatibilně (3×4 a dim U(1)×SU(2)×SU(3)) |
| odkud je 112 | „derived from the necessary degrees of freedom" | nikde není uveden počet |
| σ(W·S + b), harmonický, σ = tanh | „energy … invariant" | **2·10⁻⁵ × E(0)** po 10⁵ krocích |
| σ(W·S + b), kyvadlo, W nafitovaná na přesná data | „energy … invariant" | chyba 1 kroku 1,5·10⁻³, **10,2 × E(0)** |
| leapfrog na týchž dvou soustavách, bez trénování | — | **1,0000 × E(0)** na obou |
| symplektická podmínka při 112×112 | „preserves the Lie group symmetries" | **6216 z 12544** podmínek, nikde nenapsaných |
| „tensor product ⊗ … in a single clock cycle" | 1 takt | **33,6 taktů celé GPU**, na jeden uzel |
| bloková matice proti stavu, který kóduje | „natively processed by Tensor Cores" | stejných **56 FLOP/byte**, 112× operací a provozu |
| uzlů stavu rezidentních ve 24 GB | „millions of nodes" | **513 588** (80³), jen stav |
| uzlů, které dovolí propustnost při 60 Hz | „absolute real-time" | **310 906** (68³), nic se nekreslí |
| zatížení Tensor Cores | „efficient 15-20 % range" | 113,6–120,7 ze 142 TFLOP/s **nevyužitých** |
| „molecular brute-force simulation" | plný prostor 112D | 513 588 atomů = krychle křemíku **21,7 nm** na kartu |
| „zero-branching" | žádné if/else | vyvrací to vlastní §3 předáváním paprsek/materie |
| „eliminate hysteresis losses" tvarem | odstraněné | plocha B–H smyčky je vlastnost materiálu; tvar snižuje, neodstraňuje |

## Rozsah

Preprint je sedmistránkový koncepční whitepaper bez přiloženého kódu a bez
numerických výsledků, takže není co benchmarkovat. Dvě skutečnosti o prezentaci jsou
v §9 textu zaznamenané, ne vydávané za vady: hlavička deponovaného PDF říká
`Author: Michal Mazgal [ORCID: enter-your-orcid]` a OpenQL je už jméno kvantového
programovacího frameworku z QuTech / TU Delft
([10.1145/3474222](https://doi.org/10.1145/3474222)).

## Doprovodná vyhodnocení

- [`octonion-mppt-eval`](https://github.com/karagos01/octonion-mppt-eval) — oktonionová dvouvrstvá šablona s asociátorem a magnonová tvrzení
- [`xternary-eval`](https://github.com/karagos01/xternary-eval) — 2bitový inferenční engine pro LLM
- [`causal-trilogy-eval`](https://github.com/karagos01/causal-trilogy-eval) — trilogie CQFT / PCTP / SOTP ze září 2026

## Licence

Kód (všechny `*.py` a `run_all.sh`): MIT, viz `LICENSE`.
Text `PAPER.md` a `PAPER.cs.md`: CC BY 4.0.
