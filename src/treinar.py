import argparse
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ARTEFATO, CHAVES, DADOS_GOLD, INDICADORES

ANO_HISTORICO = 2023
ANO_ALVO = 2024
LIMIAR = 0.50


def treinar(dados: pd.DataFrame) -> dict:
    renomear = {coluna: f"{coluna}_{ANO_HISTORICO}" for coluna in INDICADORES}
    historico = dados.loc[
        dados["ano"].eq(ANO_HISTORICO), CHAVES + INDICADORES
    ].rename(columns=renomear)
    resultado = dados.loc[
        dados["ano"].eq(ANO_ALVO), CHAVES + ["taxa_oficial", "meta_do_ano"]
    ]
    painel = (
        historico.merge(resultado, on=CHAVES, validate="one_to_one")
        .dropna(subset=["taxa_oficial", "meta_do_ano"])
        .sort_values(CHAVES)
    )
    if painel[CHAVES].isna().any().any():
        raise ValueError("A chave município–rede deve ser completa.")
    alvo = painel["taxa_oficial"].lt(painel["meta_do_ano"]).astype(int)
    if alvo.nunique() != 2:
        raise ValueError("O painel rotulado deve conter as duas classes de risco.")

    numericas = list(renomear.values())
    features = numericas + ["rede"]
    preprocessador = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numericas),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), ["rede"]),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessador),
        ("model", RandomForestClassifier(
            n_estimators=80,
            max_depth=8,
            min_samples_leaf=15,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipeline.fit(painel[features], alvo)
    return {
        "pipeline": pipeline,
        "features": features,
        "limiar": LIMIAR,
        "ano_historico_treino": ANO_HISTORICO,
        "ano_alvo_treino": ANO_ALVO,
        "redes_treino": sorted(painel["rede"].unique().tolist()),
        "versoes": {
            nome: version(nome) for nome in ("scikit-learn", "numpy", "pandas", "joblib")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina o modelo municipal com a Gold.")
    parser.add_argument("--dados", type=Path, default=DADOS_GOLD)
    parser.add_argument("--modelo", type=Path, default=ARTEFATO)
    args = parser.parse_args()
    try:
        artefato = treinar(pd.read_csv(args.dados))
        args.modelo.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=args.modelo.parent, suffix=".tmp", delete=False) as arquivo:
            temporario = Path(arquivo.name)
        try:
            joblib.dump(artefato, temporario, compress=3)
            temporario.replace(args.modelo)
        finally:
            temporario.unlink(missing_ok=True)
    except (OSError, ValueError, KeyError) as erro:
        parser.exit(1, f"Erro no treinamento: {erro}\n")
    print(f"Modelo salvo em {args.modelo}")


if __name__ == "__main__":
    main()
