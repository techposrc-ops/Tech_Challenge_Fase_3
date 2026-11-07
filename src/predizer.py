"""Gera um ranking de risco de alfabetização por município e rede.

Execute a partir da raiz do projeto:

    python -m src.predizer --ano-previsao 2025 --saida resultados/risco_2025.csv

Para prever um ano, o comando usa os indicadores do ano anterior.

Por padrão, o comando usa o CSV municipal da camada Gold e o modelo
exportado por `python -m src.treinar`. Use `--dados` e `--modelo` para
indicar outros arquivos. Carregue apenas artefatos Joblib de origem confiável.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .config import ARTEFATO, CHAVES, DADOS_GOLD


def pontuar_municipios(dados: pd.DataFrame, ano_historico: int, artefato: dict) -> pd.DataFrame:
    """Estima o risco de ficar abaixo da meta no ano seguinte.

    `dados` deve conter ano, id_municipio, rede e os indicadores históricos
    exigidos pelo modelo, com os nomes originais da Gold. O lote é filtrado
    pelo ano histórico e pelas redes cobertas pelo treinamento. Cada chave
    município–rede deve ser completa e única no lote filtrado.

    `artefato` é o dicionário exportado pelo treinamento, com o pipeline
    ajustado, o esquema de atributos, o ano histórico de treino, as redes
    cobertas e o limiar de classificação.

    O resultado contém id_municipio, rede, ano_alvo, probabilidade_risco
    e risco_previsto. A classe vale 1 quando a probabilidade estimada é
    maior ou igual ao limiar e 0 nos demais casos. O ranking ordena da
    maior para a menor probabilidade e desempata por município e rede.

    Um lote vazio ou com chaves incompletas ou duplicadas gera ValueError.
    """
    # O pipeline espera os nomes usados no treino, mesmo ao receber indicadores de outro ano.
    sufixo = f"_{artefato['ano_historico_treino']}"
    renomear = {
        atributo.removesuffix(sufixo): atributo
        for atributo in artefato["features"] if atributo.endswith(sufixo)
    }
    base = dados.loc[
        dados["ano"].eq(ano_historico) & dados["rede"].isin(artefato["redes_treino"]),
        CHAVES + list(renomear),
    ].rename(columns=renomear)
    if base.empty:
        raise ValueError("Nenhum registro elegível para o ano e as redes do modelo.")
    if base[CHAVES].isna().any().any() or base.duplicated(CHAVES).any():
        raise ValueError("A chave município–rede deve ser completa e única no lote.")

    pipeline = artefato["pipeline"]
    coluna_risco = pipeline.classes_.tolist().index(1)
    probabilidades = pipeline.predict_proba(base[artefato["features"]])[:, coluna_risco]
    return base[CHAVES].assign(
        ano_alvo=ano_historico + 1,
        probabilidade_risco=probabilidades,
        risco_previsto=(probabilidades >= artefato["limiar"]).astype(int),
    ).sort_values(
        ["probabilidade_risco", *CHAVES], ascending=[False, True, True]
    ).reset_index(drop=True)


def main() -> None:
    """Lê as entradas da linha de comando e grava o ranking em CSV."""
    parser = argparse.ArgumentParser(description="Calcula o risco municipal em lote.")
    anos = parser.add_mutually_exclusive_group(required=True)
    anos.add_argument("--ano-previsao", type=int, help="Ano para o qual o risco será estimado.")
    anos.add_argument("--ano-historico", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--saida", type=Path, required=True)
    parser.add_argument("--dados", type=Path, default=DADOS_GOLD)
    parser.add_argument("--modelo", type=Path, default=ARTEFATO, help="Artefato joblib de origem confiável.")
    args = parser.parse_args()
    if args.saida.resolve() in {args.dados.resolve(), args.modelo.resolve()}:
        parser.error("--saida deve apontar para um arquivo diferente das entradas.")
    try:
        ano_historico = args.ano_historico if args.ano_previsao is None else args.ano_previsao - 1
        ranking = pontuar_municipios(
            pd.read_csv(args.dados), ano_historico, joblib.load(args.modelo)
        )
        args.saida.parent.mkdir(parents=True, exist_ok=True)
        ranking.to_csv(args.saida, index=False)
    except (OSError, ValueError, KeyError) as erro:
        parser.exit(1, f"Erro na inferência: {erro}\n")
    print(f"{len(ranking)} registros pontuados. Ranking salvo em {args.saida}")


if __name__ == "__main__":
    main()
