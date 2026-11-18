# Pipeline — AlphaEarth vs S1/S2 para delineamento de café (MG, 2017)

Código do método (versionado). Dados e saídas vão para `data/` na raiz do repo
(não-versionado).
Protocolo confirmatório (hipóteses, ordem de execução, limiares congelados) em
[`../docs/protocol/confirmatory.md`](../docs/protocol/confirmatory.md).

## Ordem

| Passo | Arquivo | Precisa EE? | O que faz |
|-------|---------|-------------|-----------|
| 1 | `01_extract_reference.py` | não | Extrai café MG 2017 do `br_conab` → GeoParquet local + área + região-proxy |
| 2 | `02_tiles.py` | não | Grade de tiles em MG + amostragem estratificada (região × tamanho × densidade) |
| 3 | `03_gee_stacks.py` | sim | **Baixa** S2/S1 **sazonais** (chuvosa/seca 2017) + AlphaEarth por tile, UTM → GeoTIFF em `data/rasters/gee/` |
| 4 | `04_probe.py` | **não (local)** | Split espacial por célula + labels (fronteira/interior) + RF idêntico por experimento → P(fronteira) |
| 5 | `05_validate.py` | não | t* em validação → watershed → métricas de objeto (IoU, Boundary F1, Clinton) + AP de fronteira + Wilcoxon no TESTE |
| 6 | `06_figures.py` | não | Figuras e tabelas derivadas das métricas já calculadas |
| 7 | `07_context_figures.py` | não | Figura de área de estudo e distribuição de área |
| 8 | `08_pipeline_figure.py` | não | Figura pictórica do desenho experimental |
| 9 | `09_context_probe.py` | não | Probes E1c–E4c (filtros de janela) e predição nos tiles confirmatórios, com checagem de reprodução |
| 10 | `10_context_eval.py` | não | Watershed e fusão de regiões (RAG), limiares da validação original, métricas estritas e testes descritivos |
| 11 | `11_confirm_tiles.py` | sim (download via 03) | Lista ordenada de tiles confirmatórios, elegibilidade, deriva dos rasters, cobertura |
| 12 | `12_confirmatory.py` | não | H1 (não superioridade), H2, H3 (TOST), gatekeeping e sensibilidades |
| 13 | `13_posthoc_checks.py` | não | Pós-hoc (revisão): seeds 1/2, E3c com PCA-16 (80 features), ablação dos filtros |

Módulos compartilhados: `probing.py` (rótulos, filtros, probes), `objects.py` (decodificadores,
métricas), `stats.py` (Hodges–Lehmann, Wilcoxon, TOST, Holm), `figstyle.py` (estilo das figuras).

## Setup

Os comandos abaixo assumem execução a partir da raiz do repositório.

```bash
python3 -m venv .venv
.venv/bin/pip install -r code/requirements.txt
# Earth Engine (passo 3) — login interativo (abre o navegador), uma vez:
.venv/bin/earthengine authenticate
# EE moderno exige um projeto Google Cloud com a Earth Engine API habilitada:
export EE_PROJECT="seu-projeto-gcp"
```

O CLI `earthengine` não fica no PATH global — está no venv (`.venv/bin/earthengine`).
Se já usa GEE, você já tem um projeto; senão, habilite a Earth Engine API em https://console.cloud.google.com.

## Rodar

```bash
.venv/bin/python code/01_extract_reference.py
.venv/bin/python code/02_tiles.py --n 80
.venv/bin/python code/03_gee_stacks.py    # BAIXA insumos (GEE -> disco)
.venv/bin/python code/04_probe.py         # probe RF LOCAL -> P(fronteira) E1-E4
.venv/bin/python code/05_validate.py      # watershed + métricas E1-E4 (teste)
.venv/bin/python code/05_validate.py --variant lin
.venv/bin/python code/06_figures.py       # figuras/tabelas derivadas das métricas
.venv/bin/python code/07_context_figures.py
.venv/bin/python code/08_pipeline_figure.py
```

## Rodar com Docker

O container substitui o `.venv` local. Os comandos abaixo assumem Docker Compose v2.
`data/` fica montado como volume do projeto e continua fora do git.

```bash
cp .env.example .env
# Edite EE_PROJECT em .env antes de rodar o passo 3.
docker compose build
docker compose run --rm pipeline python code/check_artifacts.py
```

Earth Engine dentro do Docker:

```bash
# Gera/atualiza ~/.config/earthengine/credentials pelo container.
docker compose run --rm pipeline earthengine authenticate

# Valida credenciais + EE_PROJECT sem baixar rasters.
docker compose run --rm pipeline python code/gee_check.py
```

Ordem completa:

```bash
docker compose run --rm pipeline python code/01_extract_reference.py
docker compose run --rm pipeline python code/02_tiles.py --n 80
docker compose run --rm pipeline python code/03_gee_stacks.py
docker compose run --rm pipeline python code/04_probe.py
docker compose run --rm pipeline python code/05_validate.py
docker compose run --rm pipeline python code/05_validate.py --variant lin
docker compose run --rm pipeline python code/06_figures.py
docker compose run --rm pipeline python code/07_context_figures.py
docker compose run --rm pipeline python code/08_pipeline_figure.py
```

Controles de contexto e rodada confirmatória (ordem do protocolo, seção 11):

```bash
R="docker compose run --rm pipeline python"
$R code/09_context_probe.py --scope original                       # E1c-E4c (rf, lin)
$R code/09_context_probe.py --scope original --variants rf8        # sensibilidade
for v in rf lin; do for d in ws rag; do
  $R code/10_context_eval.py --scope original --variant $v --decoder $d
done; done
$R code/12_confirmatory.py --scope original
$R code/11_confirm_tiles.py --stage drift-setup
docker compose run --rm -e ALPHAEARTH_MG_DATA=/work/data/confirm/drift pipeline \
  python code/03_gee_stacks.py --allow-partial
$R code/11_confirm_tiles.py --stage drift-compare
$R code/11_confirm_tiles.py --stage materialize                    # lista congelada -> data/confirm
docker compose run --rm -e ALPHAEARTH_MG_DATA=/work/data/confirm pipeline \
  python code/03_gee_stacks.py --allow-partial                      # até 3 tentativas
$R code/11_confirm_tiles.py --stage labels
$R code/09_context_probe.py --scope confirm
$R code/09_context_probe.py --scope confirm --exps E1c E2c E3c E4c --variants rf8
for v in rf lin; do for d in ws rag; do
  $R code/10_context_eval.py --scope confirm --variant $v --decoder $d
done; done
$R code/12_confirmatory.py --scope confirm
$R code/13_posthoc_checks.py                                       # pós-hoc, não confirmatório
make figures
```

`11 --stage sample` gerou a lista ordenada em `docs/protocol/` e recusa rodar de novo (amostra
congelada). Os limiares de `data/ctx/metrics/thresholds_*.json` são verificados por SHA-256.

Para o passo 3, o Compose monta `~/.config/earthengine` dentro do container. Você pode
autenticar no host ou pelo próprio container; nos dois casos as credenciais ficam no mesmo
diretório do host.

## Layout dos artefatos

`data/` é a raiz canônica dos artefatos regeneráveis. Para usar outro local, defina
`ALPHAEARTH_MG_DATA=/caminho/data`.

```text
data/
├── reference/ref_cafe_mg_2017.parquet
├── tiles/tiles_eval.geojson · split.json
├── rasters/gee/{tile}_{s1,s2,ae}.tif
├── labels/labels_{tile}.tif
├── probabilities/rf/prob_{tile}_{E1..E4}.tif
├── probabilities/lin/prob_{tile}_{E1..E4}_lin.tif
├── metrics/rf/metrics_E1_E4*.csv · thresholds.json
├── metrics/lin/metrics_E1_E4_lin*.csv · thresholds_lin.json
├── ctx/probabilities/{rf,lin,rf8}/prob_{tile}_{E1c..E4c}.tif
├── ctx/metrics/{ws,rag}_{rf,lin}_*.csv · thresholds_*.json · ap_per_tile.csv · confirmatory_summary.json
├── confirm/  (mesma estrutura de data/ para os tiles confirmatórios: tiles/, rasters/, labels/,
│              probabilities/, metrics/ com tile_selection.csv, raster_drift.csv e posthoc/)
└── view/*.gpkg
```

## Idempotência

Os scripts são seguros para rerun com os mesmos inputs. Outputs principais são escritos em
arquivo temporário no mesmo diretório e publicados com replace atômico só após sucesso; nos
passos com múltiplos artefatos locais (`04`–`06`), os arquivos são staged e só substituem os
outputs antigos ao fim do passo. O passo `03` mantém cache por raster: arquivos existentes e
válidos são reutilizados; cache inválido só é substituído depois de um novo download válido.

Migração do layout antigo:

```bash
.venv/bin/python code/migrate_layout.py      # dry-run
.venv/bin/python code/migrate_layout.py --apply
.venv/bin/python code/check_artifacts.py
```

## Notas de viabilidade

- **S2 em 2017:** usa **TOA** (`COPERNICUS/S2_HARMONIZED`) — a SR/L2A é incompleta antes de dez/2018 (resolvido).
- **AlphaEarth > 48 MB:** baixado em **chunks de 12 bandas** (`03`) e remontado.
- **R4 (escala de features):** o probe principal é RF (invariante a escala) — a questão de
  normalização entre fontes não afeta o resultado; probe linear (se usado) padroniza com stats do treino.
- **Referência:** `metrics:area` é nulo no parquet → área vem da geometria.
