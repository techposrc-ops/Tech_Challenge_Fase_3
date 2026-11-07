from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DADOS_GOLD = RAIZ / "data" / "gold" / "indicadores_municipio.csv"
ARTEFATO = RAIZ / "modelos" / "risco_alfabetizacao.joblib"
CHAVES = ["id_municipio", "rede"]
INDICADORES = [
    "taxa_oficial",
    "alunos_presentes",
    "percentual_participacao",
    "nivel_alfabetizacao",
    "media_proficiencia",
    "media_portugues",
]
