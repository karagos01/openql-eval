# Polovina titulu je vyloučená teorémem: vyhodnocení openQL / openOL

**karagos01**, říjen 2026

Nezávislé vyhodnocení práce M. Mazgal, *openQL / openOL: Tensor Matrix
Architecture for Unified Simulation of Physical Fields in 12D and 112D via GPU
Tensor Cores*, Zenodo, 3. října 2026, DOI
[10.5281/zenodo.23112560](https://doi.org/10.5281/zenodo.23112560) (CC BY 4.0).

> Komentáře v kódu a výpisy skriptů jsou anglicky. Anglická verze textu je
> v [`PAPER.md`](PAPER.md).

## Abstrakt

Preprint navrhuje nahradit polygonová grafická API volumetrickou architekturou, ve
které každý voxel nese reálnou blokovou matici 12×12 (openQL) nebo 112×112
(openOL), získanou „by transforming quaternion and octonion algebra into unified
block matrices", a ve které časový vývoj běží na Tensor Cores jako symplektický
integrátor. Toto vyhodnocení kontroluje ty části, které spočítat lze.

Kvaternionová polovina je správná a je provedená celá: matice 4×4 vytištěná v §2.1
je levá regulární reprezentace ℍ a splňuje M(q)M(r) = M(qr) na 1,8·10⁻¹⁵. Je to
poprvé, kdy centrální algebraický krok v preprintech tohoto autora obstojí od
začátku do konce, a doprovodná reference je citovaná správně.

Oktonionová polovina je vyloučená teorémem, ne inženýrskou obtížností. Násobení
matic je asociativní, takže každý homomorfismus algeber φ: 𝕆 → Mₙ(ℝ) musí anulovat
každý asociátor; asociátory autorovy vlastní publikované násobicí tabulky generují
celou sedmirozměrnou imaginární část 𝕆, takže φ zachová 1 z 8 dimenzí a — protože
𝕆 je jednoduchá algebra — je nulové zobrazení pro každé n. Tatáž konstrukce, která
je pro ℍ exaktní, je pro 𝕆 chybná o relativních 2,8. Jediná citace práce, která
oktoniony pokrývá, Baezův přehledový článek, uvádí nonasociativitu na své první
stránce.

Změřeny jsou tři další vady. Uzel 12×12 je blokově diagonální, takže jeho algebra
je ℍ⊕ℍ⊕ℍ: perturbace jednoho bloku nepohne ničím mimo něj, 96 ze 144 prvků je
strukturně nulových a „unified state structure" ukládá 12 reálných čísel do 144
slotů. Předpis S(t+1) = σ(W·S(t) + b) není symplektický — když se metodou nejmenších
kvadrátů nafituje na přesná jednokroková data z kyvadla, je po jednom kroku přesný
na 1,5·10⁻³ a po dlouhém rolloutu skončí na 10,2× počáteční energii, zatímco
leapfrog bez jediné váhy a bez trénování drží 1,0000. A bloková maticová
reprezentace leží na roofline přesně ve stejném bodě jako ten 112složkový stav,
který kóduje — 56 FLOP/byte proti hřebenu 152 — a stojí přitom 112× více paměťového
provozu a 112× více operací; jedna 24GB karta udrží 513 588 uzlů samotného stavu a
nic jiného a propustnost dovolí mřížku 68³ při 60 Hz, aniž by se vykreslil jediný
pixel.

Všechno níže reprodukuje `./run_all.sh` (čistě CPU, bez sítě, ~20 s).

## 1. Co je správně

**Kvaternionová matice 4×4.** §2.1 tiskne pro q = a + bi + cj + dk

```
M(q) =  [ a  -b  -c  -d ]
        [ b   a  -d   c ]
        [ c   d   a  -b ]
        [ d  -c   b   a ]
```

a nazývá ji izomorfní. Je. `quaternion.py` spočítá Hamiltonův součin nezávisle a
ověří M(q)M(r) = M(qr) na 5000 náhodných párech: maximální absolutní chyba
**1,78·10⁻¹⁵**, **1,51·10⁻¹⁶** relativně k velikosti prvků. Je to standardní levá
regulární reprezentace ℍ, znaménkové konvence ve vytištěné matici jsou správné a
reference [1] (Kuipers) je na to korektní zdroj.

To má význam nad rámec jedné rovnice. V dřívějších preprintech tohoto autora byla
algebra buď chybná, nebo chyběla; v září v trilogii byly oktonionové strukturní
konstanty správné, ale fyzikální mapování ne. Tady je jedno celé algebraické
tvrzení vysloveno a ověřeno. Je potřeba to říct přímo a jako první.

**Velikost uzlu je uvedená poctivě.** §4 říká „the memory footprint of a single
node reaches tens of kilobytes". Matice 112×112 ve fp16 má 24,5 KiB. Práce svou
vlastní cenu nezmenšuje.

**Reference existují.** Je jich pět a všech pět existuje a jsou to nosné práce ve
svých oborech: Kuipers na kvaterniony, Raissi a kol. na PINNs, Markidis a kol. na
programovatelnost Tensor Cores, Mildenhall a kol. na NeRF, Baez na oktoniony.
Dřívější MPPT a CQFT/PCTP neměly bibliografii vůbec. §9 níže je o tom, co ty
reference mají nést, což je jiná otázka než jestli existují.

## 2. Oktonionová polovina titulu nemůže existovat

Titul i abstrakt říkají, že architektura funguje „by transforming quaternion **and
octonion** algebra into unified block matrices", a engine jménem openOL je ten
oktonionový. V těle práce není oktonionové násobení nikde. Nemůže být, a důvod je
teorém.

Násobení matic je asociativní. Takže pro každý homomorfismus algeber φ: 𝕆 → Mₙ(ℝ)
a každá x, y, z platí

    φ([x,y,z]) = φ((xy)z) − φ(x(yz)) = (φ(x)φ(y))φ(z) − φ(x)(φ(y)φ(z)) = 0,

tedy každý asociátor leží v ker φ. `octonion.py` změří, co ten kernel obsahuje, na
autorově vlastní násobicí tabulce — orientovaných Fanových triádách z SOTP Eq. (3),
o kterých předchozí vyhodnocení ověřilo, že dávají platnou normovanou algebru
s dělením:

| | naměřeno |
|---|---|
| bázové trojice (i,j,k) s [eᵢ,eⱼ,eₖ] ≠ 0 | **168 z 512** |
| hodnost obalu těch asociátorů | **7** |
| dim Im 𝕆 | 7 |
| reálná složka dosažená jakýmkoli asociátorem | 0,0 |

Asociátory generují imaginární část přesně. Takže ker φ ⊇ Im 𝕆, φ zachová **1 z 8
dimenzí (12,5 %)**, a protože 𝕆 je jednoduchá algebra, musí být kernel buď 0, nebo
celé 𝕆 — je celé 𝕆 a φ je nulové zobrazení. Platí to pro každou velikost matice,
nad každým tělesem, při každé volbě báze a při každém blokování, takže se k tomu
žádným inženýrstvím nedostaneš.

Totéž změřené místo odvozeného: konstrukce, která funguje pro ℍ, je levá regulární
reprezentace L_x: y ↦ xy, která je pro 𝕆 úplně dobrá reálná matice 8×8 — jen není
multiplikativní. Na 5000 náhodných párech pro každou:

| algebra | max \|L_x L_y − L_{xy}\| | relativně |
|---|---|---|
| ℍ (4×4) | 1,78·10⁻¹⁵ | 4,24·10⁻¹⁶ |
| 𝕆 (8×8) | **29,5** | **2,76** |

A selhává to právě tam, kde to architektura potřebuje. 𝕆 je alternativní, takže
L_x L_x = L_{xx} platí exaktně (naměřeno 0,0 na bázových jednotkách) a každý
smíšený součin selže (2,0 na bázových jednotkách). MMA pipeline nad „unified"
uzlem počítá smíšené součiny; to je celý důvod, proč se několik fyzikálních
veličin dává do jedné matice.

Reference [5] je J. C. Baez, *The Octonions*, Bull. AMS **39** (2002) 145–205,
citovaná na „underlying Lie algebra and gauge group symmetries corresponding to
the 12D/112D spatial mappings". Ten přehledový článek neobsahuje žádné
dvanáctirozměrné ani 112rozměrné mapování a nonasociativitu 𝕆 uvádí ve své úvodní
části. Jediný zdroj citovaný k oktonionové polovině architektury je ten zdroj,
který oktonionovou polovinu vylučuje.

## 3. Uzel 12D jsou tři kvaterniony, které se nemají jak potkat

§2.1 staví uzel „concatenating three such state matrices" a dostává „a unified
state structure (voxel) of a circuit or matter … enabling computation via a single
Matrix Multiply-Accumulate (MMA) instruction". Tři bloky 4×4 jsou jediné čtení,
které z nich dá matici 12×12, a vrstva 12D matici 12×12 potřebuje. `quaternion.py`
měří, co to je za objekt:

| | naměřeno |
|---|---|
| součin se rovná blokově (qᵢrᵢ) | chyba 4,4·10⁻¹⁶ |
| perturbace bloku 0 operandu pohne blokem 0 o | 1,940 |
| …a vším mimo blok 0 o | **0,000** |
| strukturně nulové prvky | **96 ze 144 (67 %)** |
| prvky vázající blok i na blok j | **0** |
| nesený stav / obsazené sloty | 12 / 144 (**12× redundance**) |

Algebra je ℍ⊕ℍ⊕ℍ. Tři kvaterniony se vyvíjejí vedle sebe a nemají kanál, kterým by
si cokoli předaly, takže uzel jsou tři nezávislé čtyřrozměrné soustavy v nádobě
dvanáctkrát větší, než je jejich stav. Ať „unified simulation of physical fields"
znamená cokoli, tohle to není: vázaný elektromagneticko-tepelně-mechanický voxel je
přesně soustava, ve které perturbace jedné části pohne ostatními, a naměřená
odezva mimo blok je identicky nulová.

Dvanáctka je navíc v jedné a téže práci odvozená dvakrát, nekompatibilně. §2.1 ji
staví jako 3 kvaterniony × 4 složky. §6 směruje vývoj k „the fundamental gauge
group U(1) × SU(2) × SU(3)" a §2.3 mluví o „the macroscopic 12D gauge groups";
dim U(1) + dim SU(2) + dim SU(3) = 1 + 3 + 8 = 12. Jsou to různé dvanáctky — tři
kopie čtyřrozměrné algebry proti Lieově algebře dvanáctirozměrné grupy, která
žádnou kvaternionovou blokovou strukturu nemá — a nic v práci je nespojuje.

Číslo 112 nemá odvození žádné. §2.3 říká, že „is derived from the necessary degrees
of freedom to encapsulate the macroscopic 12D gauge groups alongside the microstate
variables of crystallography, spin configurations, and valence orbital interactions
required for phase-transition dynamics", bez jediného počtu. 112 = 28 × 4 = 14 × 8;
práce nezmiňuje ani jeden z těch rozkladů, takže není co auditovat.

## 4. σ(W·S + b) není symplektický integrátor

§2.3 je nejsilnější fyzikální tvrzení práce: engine „employs a symplectic
integration scheme directly within the matrix multiplication", předpis je

    S(t+1) = σ( W_PINN ⊗ S(t) + b )

a tím se „preserves the Lie group symmetries of the 12D/112D space, ensuring that
energy and momentum remain invariant without requiring floating-point correction
loops". `symplectic.py` to testuje třemi způsoby.

**Izolovat σ a b.** Vezmi harmonický oscilátor a zvol W = exp(dt·J), což je přesně
symplektická matice, takže není co svádět na špatný fit. 10⁵ kroků, dt = 0,01,
E(0) = 0,5:

| předpis | E(konec) | E/E(0) |
|---|---|---|
| leapfrog, bez trénování | 0,499991 | 1,00 |
| W ∈ Sp(2), σ = id, b = 0 | 0,500000 | 1,00 |
| W ∈ Sp(2), σ = id, **b = 10⁻³** | 0,547046 | **1,09** |
| W ∈ Sp(2), **σ = tanh**, b = 0 | 1,00·10⁻⁵ | **2·10⁻⁵** |
| W ∈ Sp(2), **σ = relu**, b = 0 | 2,27·10⁻⁵ | **4,5·10⁻⁵** |

Jediný řádek, který zachovává, je ten, kde σ je identita a b je nula — tedy kde je
předpis lineární symplektické zobrazení a ne vrstva neuronové sítě. Nelinearita,
kterou práce předepisuje, stojí pět řádů energie; bias 10⁻³ přidá 9 %.

**Spočítat podmínky.** Symplektičnost je WᵀJW = J, uzavřená podmínka kodimenze
dim GL − dim Sp:

| 2n | dim GL(2n) | dim Sp(2n) | podmínek | podíl |
|---|---|---|---|---|
| 12 | 144 | 78 | 66 | 45,8 % |
| 112 | 12544 | 6328 | **6216** | **49,6 %** |

Při 112×112 musí přibližně polovina matice vah splňovat rovnosti, které práce
nikde nepíše, a nic v PINN ztrátě na trajektoriích je nevynucuje. Náhodná matice
112×112 je od té podmínky vzdálená max |WᵀJW − J| = 1,19. Sp(112) má v GL(112)
nulovou míru, takže „preserves the Lie group symmetries" je vlastnost, kterou
naučená W má s pravděpodobností nula.

**Natrénovat to tak, jak práce navrhuje, a pak změřit.** Nafituj W a b metodou
nejmenších kvadrátů na 20 000 přesných jednokrokových párů ze skutečného
hamiltonovského toku a pak to rozjeď na 10⁵ kroků:

| soustava | předpis | chyba 1 kroku | \|WᵀJW − J\| | E(10⁵) | E/E(0) |
|---|---|---|---|---|---|
| harmonická | leapfrog, bez trénování | — | — | 1,12498 | **1,0000** |
| harmonická | σ = id | 4,4·10⁻¹⁶ | 3,3·10⁻¹⁶ | 1,12498 | **1,0000** |
| harmonická | σ = tanh | 4,4·10⁻² | 2,7·10⁻¹ | 2,73851 | **2,4342** |
| kyvadlo | leapfrog, bez trénování | — | — | 1,41612 | **1,0000** |
| kyvadlo | σ = id | 1,5·10⁻³ | 2,5·10⁻⁵ | 14,4747 | **10,2212** |
| kyvadlo | σ = tanh | 4,4·10⁻² | 2,7·10⁻¹ | 2,46981 | **1,7440** |

U harmonického oscilátoru *je* přesné jednokrokové zobrazení lineární a
symplektické, takže ho nejmenší kvadráty zrekonstruují na 4·10⁻¹⁶ a energie se
zachová. To je ten speciální případ, ne funkční architektura — a když se přidá
nelinearita, kterou práce předepisuje, tatáž soustava získá faktor 2,4. U kyvadla
nezachovává nic: lineární fit je po jednom kroku přesný na 1,5·10⁻³, což je dobrý
fit, a skončí desetkrát příliš energetický. Leapfrog drží obojí na pět desetinných
míst, bez vah a bez trénování.

Malé jednokrokové reziduum není zachování. Zachování je vlastnost struktury
zobrazení a σ(W·x + b) tu strukturu nemá pro žádné W a b. Věta „without requiring
floating-point correction loops" to má obráceně: korekční smyčky jsou to, co
potřebuje nesymplektické schéma, a navržené schéma je nesymplektické.

K notaci: ⊗ v té rovnici nemůže být tenzorový součin. Kroneckerův součin dvou matic
112×112 má 157 351 936 prvků, 300 MiB na uzel ve fp16. Čteno jako maticový součin,
který práce myslí, je jedno W·S 2 809 856 operací; při 142 TFLOP/s fp16 a boostu
1,70 GHz na RTX 3090 dá celý čip 83 529 operací za takt, takže jeden uzel zabere
**33,6 taktů celé GPU**, ne „a single clock cycle".

## 5. Bloková matice je 112× režie při stejné aritmetické intenzitě

Tohle je tichá vada a zároveň ta, která rozhoduje, jestli by architektura mohla
fungovat vůbec. Fyzika v uzlu openOL je 112 čísel — autorův vlastní výčet
potenciálů, spinů, mřížkových a valenčních proměnných. Uzel je ukládá jako matici
112×112:

| | |
|---|---|
| bloková matice 112×112 ve fp16 | 24,5 KiB |
| těch 112 fyzikálních potenciálů | 0,219 KiB |
| režie | **112×** |
| podíl nenulových při 28 kvaternionových blocích 4×4 | 3,57 % |

Zdůvodnění, proč to platit, je propustnost Tensor Cores: matice jsou „a format
natively processed by Tensor Cores". Jenže poloha na roofline se reprezentací
nemění:

| operace | FLOP | bytů | FLOP/byte |
|---|---|---|---|
| W·S, uzel 112×112 | 2 809 856 | 50 176 | **56,0** |
| W·s, vektor 112 | 25 088 | 448 | **56,0** |

Totožná aritmetická intenzita, 112× operací a 112× provozu. Obojí leží *pod*
hřebenem RTX 3090 na 152 FLOP/byte, takže obojí je memory-bound, a povýšení stavu
na matici nepřibližuje k compute-bound režimu, kde se Tensor Cores vyplatí.
Násobí práci 112× a nechává úzké hrdlo přesně tam, kde bylo.

Tím se taky vysvětluje vlastní číslo z §3. Práce uvádí, že pipeline „keeps Tensor
Core load in the efficient 15-20 % range". Při špičce 142 TFLOP/s je to 21,3–28,4
TFLOP/s doručených a 113,6–120,7 TFLOP/s nevyužitých. Patnáct až dvacet procent
využití jediné jednotky, kolem které je celá architektura postavená, je symptom, ne
cíl návrhu: §2 přesouvá všechnu fyziku na Tensor Cores právě proto, aby se
nasytily. Měření výše dává důvod, proč se nenasytí — při 56 FLOP/byte uzel čeká na
VRAM, ať se práce rozvrhne jakkoli.

## 6. Paměť a kroková frekvence

Jen dvojitě bufferovaný stav — žádné matice vah, žádný framebuffer, žádný rendering
— proti 936 GB/s a 24 GB RTX 3090:

| mřížka | uzlů | S, S′ | × 24 GB | karet | s/krok | Hz |
|---|---|---|---|---|---|---|
| 32³ | 32 768 | 1,5 GiB | 0,1× | 1 | 0,002 | 569 |
| 64³ | 262 144 | 12,2 GiB | 0,5× | 1 | 0,014 | 71 |
| 80³ | 512 000 | 23,9 GiB | 1,0× | 1 | 0,027 | 36 |
| 100³ | 1 000 000 | 46,7 GiB | 1,9× | 2 | 0,054 | **19** |
| 128³ | 2 097 152 | 98,0 GiB | 4,1× | 5 | 0,112 | 8,9 |
| 256³ | 16 777 216 | 784,0 GiB | 32,7× | 33 | 0,899 | 1,1 |

Uzlů, jejichž samotný stav zaplní 24 GB: **513 588** (80³). Uzlů, které dovolí
propustnost při 60 Hz: **310 906** (68³). Takže „millions of nodes" z §3 stojí
47 GiB na milion jen za stav a realtimeový cíl „photorealistic synthetic worlds"
z §5 je krychle o 68 voxelech, na které se nic nekreslí.

§5 dál navrhuje „searching for superconductors and novel alloy phase structures in
the full 112D space of openOL via molecular brute-force simulation". Při jednom
uzlu na atom udrží 24 GB 513 588 atomů — krychli krystalického křemíku o hraně
**21,7 nm**, jen stav. To je legitimní velikost pro atomistickou práci, a právě
proto je ta srovnávací nevýhodná: klasická molekulární dynamika na tu škálu dnes
dosahuje za několik desítek bajtů na atom, ne za 24,5 KiB, a bez tvrzení o 112
dimenzích. Na jeden mol uzlů by bylo potřeba 1,2·10¹⁸ karet.

## 7. „Zero-branching" si vyvrací vlastní sekce

Vlastnost „zero-branching" je deklarovaný inženýrský přínos práce: „Solids,
liquids, and electrical currents are computed by the identical hardware logic
without any if/else statements." O dva odstavce dál, v Projection Phase:
„Movement through empty space is rapidly handled by CUDA cores; upon intersection
with matter, the computation of refraction, reflection, or interference is handed
back to the Tensor Cores."

To je branch. Je to per-ray branch závislý na datech, což je na GPU ten drahý
druh: vlákna ve warpu, která narazí na materii v různé hloubce, divergují a warp
platí za každou prošlou cestu. Raymarching do řídkého objemu je jeden
z nejdivergentnějších workloadů v grafice. Architektura branching neodstraňuje;
přesouvá ho z fyzikálního kernelu do smyčky nad paprsky, kde stojí víc.

## 8. Hystereze je vlastnost materiálu, ne tvaru

§5 navrhuje „replacing traditional BLDC motor design with evolutionary iteration in
openQL" a výsledkem mají být „optimized, organically shaped 3D fractal models of
stators and rotors that eliminate hysteresis losses".

Hysterezní ztráta na jednotku objemu a cyklus je plocha uzavřená B–H smyčkou
materiálu a celková ztráta je ta plocha krát frekvence krát objem jádra. Geometrie
do toho vstupuje jen přes indukci, na kterou je materiál vybuzen, a přes objem
přítomného materiálu, takže optimalizace tvaru hysterezní ztrátu snížit umí a
rutinně se k tomu používá. Odstranit ji vyžaduje změnit materiál — slitinu s nižší
koercitivitou, amorfní nebo nanokrystalickou pásku, ferit — nebo feromagnetické
jádro vynechat, jako u bezželezných strojů, což tu ztrátu vymění za mnohem větší
magnetizační proud. Žádný tvar, fraktální ani jiný, nenastaví plochu B–H smyčky na
nulu. Totéž splynutí vlastnosti materiálu s počitatelnou geometrií bylo
i v hysterezní argumentaci preprintu MPPT.

## 9. Jméno a prezentace

Dvě faktické poznámky, ani jedna se netýká fyziky.

**openQL je existující projekt.** OpenQL je kvantový programovací framework
z QuTech / TU Delft, publikovaný v ACM Transactions on Quantum Computing
([10.1145/3474222](https://doi.org/10.1145/3474222), preprint
[arXiv:2005.13283](https://arxiv.org/abs/2005.13283)), s veřejnou dokumentací
a repozitářem. To jméno je v užívání pro jiný druh kompilátoru v sousedním oboru.

**Autorská šablona nebyla dovyplněná.** Hlavička deponovaného PDF říká
`Author: Michal Mazgal [ORCID: enter-your-orcid]`. Ten placeholder je ve
zveřejněném záznamu.

**Co ty reference nesou.** Všech pět existuje a čtyři jsou citované na tvrzení,
která unesou: Kuipers na kvaternionovou matici, Raissi a kol. na kódování fyziky do
vah sítě, Markidis a kol. na propustnost MMA, Mildenhall a kol. na volumetrický
rendering bez polygonů. Pátá, Baez, je citovaná na mapování 12D/112D, která v ní
nejsou, a je to text, který vylučuje centrální operaci §2 (viz §2). Citovat reálné
zdroje je proti dřívějším preprintům změna; citovat je na to, co obsahují, je krok,
který zbývá.

## 10. Co se změnilo

Proti dřívějším preprintům tohoto autora se dvě věci zlepšily a jedna ne.

Zlepšilo se: poprvé existuje bibliografie a jeden centrální algebraický krok —
reprezentace ℍ maticemi 4×4 — je zároveň správný a provedený celý. V září v trilogii
byla algebra správná, ale žádná derivace nebyla dotažená; v MPPT nebylo
oktonionové mapování vůbec kontrolovatelné. §1 tohoto vyhodnocení je delší než
odpovídající sekce předchozích dvou.

Nezměnilo se: struktura argumentu. Vysloví se správný algebraický fakt, přilepí se
na něj druhý, který je vyloučený teorémem, a nad to se navrství hardwarová tvrzení,
která padnou na jednom dělení. V MPPT byla správná část Fanova rovina a vyloučená
část označování bázových jednotek fyzikálními dimenzemi; tady je správná část
kvaternionová matice a vyloučená část maticová forma pro 𝕆.

Jeden detail stojí za zaznamenání sám pro sebe. Je to první autorův deposit, který
přilepuje Maxwella na kvaterniony, a ne na oktoniony: §2.1 popisuje čtyřrozměrnou
vrstvu jako „quaternion mechanics (for computing phase shifts, rotations, and
Maxwell's equations)". Z devíti depositů připisují tři inženýrské preprinty (MPPT
v1.0 a v1.1, akustický projektor, X-Ternary) Maxwellovi „original hypercomplex
architecture" postavenou na oktonionech a trilogie Maxwella nezmiňuje vůbec.
Kvaterniony jsou ta algebra, u které historické tvrzení aspoň něco drží — §§618–619
*Treatise* z roku 1873, dvě stránky hamiltonovské operátorové notace, ve kterých
Maxwell nikdy nevynásobí dva kvaterniony — a oktoniony se u Maxwella neobjevují
nikde. Atribuce je tady pořád bez citace a pořád bez mechanismu, ale míří poprvé na
správnou algebru.

## 11. Omezení

Toto vyhodnocení je vyhodnocení sedmistránkového koncepčního whitepaperu bez
přiloženého kódu, bez rovnic nad rámec citovaných a bez numerických výsledků. Není
co spustit proti autorově vlastní implementaci, protože žádná není, takže §§5–6 jsou
aritmetika nad publikovanými hardwarovými specifikacemi, ne benchmarky. Referenční
karta je RTX 3090, kterou jmenují ostatní autorovy preprinty; na H100 (3,35 TB/s,
990 TFLOP/s fp16) vyjde hřeben na 295 FLOP/byte, uzel zůstává memory-bound na
56 FLOP/byte a čísla z §6 se s kartou škálují: rezidentní uzly o 80/24 = 3,3× a
kroková frekvence o 3,35/0,936 = 3,6×. Závěry §§5–6 na volbě karty nezávisí.

§3 čte „concatenating three such state matrices" jako blokově diagonální skládání.
Je to jediné čtení, které ze tří matic 4×4 dá objekt 12×12, a vrstva 12D objekt
12×12 potřebuje, ale práce to skládání nikde nezapisuje. Pokud bylo myšleno jiné,
vázající skládání, nebyl by výsledek reprezentace ℍ⊕ℍ⊕ℍ — a nebyl by ani
reprezentace žádné kompoziční algebry, protože Hurwitzův teorém žádnou
dvanáctirozměrnou nenechává.

Teorém z §2 se týká homomorfismů algeber. Oktoniony se samozřejmě *uložit* do matic
dají a oktonionové násobení se *spočítat* na GPU dá kontrakcí strukturního tenzoru
8×8×8, což autorův vlastní kód k MPPT dělá. Co existovat nemůže, je maticová forma,
ve které maticové násobení provádí oktonionové násobení — a právě to „transforming
octonion algebra into unified block matrices" tak, aby výsledek byl „natively
processed by Tensor Cores", vyžaduje.

## 12. Dostupnost dat

Všechna čísla v tomto textu vyrábí `./run_all.sh` v tomto repozitáři:
`quaternion.py` (§§1, 3), `octonion.py` (§2), `symplectic.py` (§4), `roofline.py`
(§§5, 6). Čistě CPU, bez sítě, asi 20 sekund. `fano.py` je násobicí tabulka
z doprovodného repozitáře `causal-trilogy-eval`, nezměněná, takže §2 měří autorovy
vlastní strukturní konstanty. Celý výstup jednoho běhu je
v [`results.log`](results.log).

## Reference

- M. Mazgal, *openQL / openOL: Tensor Matrix Architecture for Unified Simulation of Physical Fields in 12D and 112D via GPU Tensor Cores*, Zenodo, 2026. DOI [10.5281/zenodo.23112560](https://doi.org/10.5281/zenodo.23112560)
- J. C. Baez, *The Octonions*, Bulletin of the AMS **39** (2002) 145–205. DOI [10.1090/S0273-0979-01-00934-X](https://doi.org/10.1090/S0273-0979-01-00934-X)
- A. Hurwitz, *Über die Composition der quadratischen Formen von beliebig vielen Variabeln*, Nachr. Ges. Wiss. Göttingen (1898) 309–316
- J. B. Kuipers, *Quaternions and Rotation Sequences*, Princeton University Press, 1999
- M. Raissi, P. Perdikaris, G. E. Karniadakis, *Physics-informed neural networks*, J. Comput. Phys. **378** (2019) 686–707. DOI [10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)
- E. Hairer, C. Lubich, G. Wanner, *Geometric Numerical Integration*, 2. vydání, Springer, 2006
- N. Khammassi, I. Ashraf, J. van Someren, R. Nane, A. M. Krol, M. A. Rol, L. Lao, K. Bertels, C. G. Almudever, *OpenQL: A Portable Quantum Programming Framework for Quantum Accelerators*, ACM Trans. Quantum Computing **3** (2022). DOI [10.1145/3474222](https://doi.org/10.1145/3474222)
- S. Williams, A. Waterman, D. Patterson, *Roofline: an insightful visual performance model*, Comm. ACM **52** (2009) 65–76. DOI [10.1145/1498765.1498785](https://doi.org/10.1145/1498765.1498785)
- J. C. Maxwell, *A Treatise on Electricity and Magnetism*, Clarendon Press, 1873, §§618–619
