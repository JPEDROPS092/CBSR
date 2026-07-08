# DermaTriage — SBESC 2026

Pipeline clássico de visão computacional para triagem dermatológica em
dispositivos de borda. Este pacote reorganiza o notebook original em três
módulos, adiciona **salvamento de modelos** e inclui um **script de inferência
para hardware de borda** (Raspberry Pi, Jetson, mini-PC ou laptop).

## Arquivos

| Arquivo | Papel |
|---|---|
| `dermatriage_pipeline.py` | Segmentação + extração de features + `MetadataEncoder`. **Usado por treino E borda** — garante features idênticas nos dois lados. |
| `train_export.py` | Treino, validação cruzada por paciente, testes estatísticos e **exportação dos modelos** (`.joblib`). |
| `edge_infer.py` | **Inferência e benchmark no dispositivo de borda.** Predição por imagem, questionário clínico, latência/FPS/energia reais. |
| `requirements_train.txt` / `requirements_edge.txt` | Dependências de cada lado. |

## Por que dois ambientes compartilham um módulo

O erro mais comum ao levar um modelo para produção é extrair as features de um
jeito no treino e de outro na borda — o modelo "quebra" silenciosamente. Aqui
`segment()`, `extract_image_features()` e `MetadataEncoder` vivem em **um único
arquivo** importado pelos dois lados. A ordem das colunas de metadados é
**congelada no `fit()`** e salva dentro do bundle, então o vetor montado no
Raspberry Pi bate exatamente com o `StandardScaler` e o classificador.

## Treinar e exportar (Colab / Kaggle / PC com o dataset)

```bash
pip install -r requirements_train.txt
python train_export.py                      # baixa o PAD-UFES-20 via kagglehub
# ou:
python train_export.py --data /caminho/pad-ufes-20
```

Gera em `results_final/`:
- `model_binary.joblib`  → rastreio de câncer (deploy este)
- `model_6class.joblib`  → diagnóstico de 6 classes
- `models_manifest.json` → o que há em cada bundle
- figuras e tabelas do artigo (`fig*.png`, `table_*.csv`)

Por padrão exporta a configuração **Color+Meta** (a recomendada pelo artigo:
os metadados dominam e a textura não compensa o custo). Como esse modelo **não
usa textura, o dispositivo de borda dispensa o scikit-image**. Para exportar
outra configuração: `--export-config All`.

## Rodar na borda

Copie para o dispositivo **apenas 3 arquivos**: `edge_infer.py`,
`dermatriage_pipeline.py` e o `.joblib` escolhido.

```bash
pip install -r requirements_edge.txt

# verificar a instalação (sem modelo, sem dataset)
python3 edge_infer.py --selftest

# prever com imagem + metadados de um JSON
python3 edge_infer.py --model model_binary.joblib --image lesao.jpg --meta paciente.json

# preencher o questionário clínico na hora
python3 edge_infer.py --model model_binary.joblib --image lesao.jpg --questionnaire

# benchmark real de latência / FPS / energia neste hardware
python3 edge_infer.py --model model_binary.joblib --benchmark --out bench.json
```

### Energia
A latência medida é **real**. A energia por inferência é uma **estimativa**
(`latência × potência assumida`). Para o número do artigo, meça a potência
média com um wattímetro USB/inline e passe `--power <watts>`. Sem `--power`, o
script usa um valor típico por dispositivo (ex.: Raspberry Pi 4 ≈ 3,4 W).

### Formato do `paciente.json`
```json
{"age": 67, "diameter_1": 12, "diameter_2": 9, "fitspatrick": 2,
 "gender": "MALE", "region": "FACE", "itch": 1, "grew": 1, "hurt": 0,
 "changed": 1, "bleed": 1, "elevation": 1, "smoke": 1, "drink": 0,
 "pesticide": 0, "skin_cancer_history": 1, "cancer_history": 0, "biopsed": 0}
```
Campos ausentes recebem defaults (idade=60, sintomas=0).

## Aviso
Ferramenta de **triagem/auxílio**, não é diagnóstico. Casos positivos ou
incertos devem ser encaminhados a um especialista.
