# Eficiência de Recursos nas PME Europeias

Projeto final de PACD (Projetos Avançados em Ciência de Dados) — análise dos fatores determinantes da adoção de práticas de eficiência de recursos em PME europeias.

## Dados

**Flash Eurobarometer 549** (Junho 2024) — GESIS ZA8869
13.559 PME de 27 países da UE + Reino Unido, inquérito estratificado com pesos pós-estratificação (`w1_sme`).

## Estrutura do Projeto

```
code/
├── data/
│   └── raw/                # Dados originais (.sav)
├── notebooks/
│   ├── NB01_understanding.ipynb   # Carregamento, filtros, seleção de variáveis
│   └── NB02_analise.ipynb         # EDA, tratamento, MCA, modelação
├── reports/
│   ├── mca_regressions_report.md  # Relatório MCA + regressões individuais
│   └── figures/                   # Gráficos gerados
├── requirements.txt
└── README.md
```

## Notebooks

| Notebook | Conteúdo | Estado |
|----------|----------|--------|
| **NB01** | Carregamento SPSS, recoding DK/NA, filtro geográfico (EU+UK), filtro PME (<250 FTE), seleção de variáveis | Concluído |
| **NB02** — Secção 2 | AED: distribuições, bivariadas VD×VI, testes estatísticos, colinearidade | Concluído |
| **NB02** — Secção 3 | Tratamento: NAs, outliers, dummies, variáveis derivadas, MCA (VD alternativa) | Concluído |
| **NB02** — Secção 4.1 | Regressões logísticas individuais (9 práticas Q1), comparação com pesos MCA | Concluído |
| **NB02** — Secção 4.2+ | Modelo multinível (HLM) com variáveis de nível país | Pendente |

## Variáveis Principais

### Variável Dependente
- **`intensity_index`** (0–9) — soma das 9 práticas Q1 adotadas
- **`mca_score`** — score MCA ponderado pela raridade (VD primária)

### Preditores de Nível 1 (empresa)
- Setor (`nace_b`), dimensão (`scr10`), antiguidade (`scr12`)
- Evolução do volume de negócios (`scr13a`), evolução do emprego (`scr11a`)
- Volume de negócios (`scr14`), impacto nos custos de produção (`q3`)
- Investimento ambiental (`q4`), dificuldade financeira (`scr11b`)
- Barreiras à eficiência (`q7.1`–`q7.12`)

### Preditores de Nível 2 (país) — pendentes
- Legislação ambiental, indicadores económicos, índices contextuais

## Metodologia

- **CRISP-DM** como framework de trabalho
- **MCA** (Análise de Correspondências Múltiplas) para ponderar as práticas Q1
- **Modelos multinível** (HLM) para acomodar a estrutura hierárquica (ICC país ≈ 13%)
- Linguagem: **Português Europeu** em todos os notebooks

## Setup

```bash
git clone https://github.com/rapsagg/pacd && cd pacd/code
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Colocar initial_data.sav em data/raw/
jupyter notebook notebooks/
```
