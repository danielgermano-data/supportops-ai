# Competências demonstradas e próximos passos

SupportOps AI é um projeto de estudo para posições de entrada em Machine Learning e engenharia de IA aplicada. Seu valor em entrevista depende de compreender e conseguir modificar a implementação. A construção contou com assistência de IA; publicar o repositório não comprova domínio individual de todo o código.

## O que pode ser demonstrado agora

| Competência | Evidência no projeto | Demonstração em entrevista |
|---|---|---|
| Preparação de dados e prevenção de vazamento | `scripts/train.py`, `reports/splits.json` | Explicar por que treino, validação e teste têm funções diferentes e como as duplicatas foram tratadas. |
| Treinamento supervisionado | TF-IDF + regressão logística em 77 classes | Alterar um hiperparâmetro usando a validação e discutir o efeito, sem escolher pela métrica do teste. |
| Avaliação de ML | Baseline, macro-F1, top-3, matriz de confusão e rejeição | Mostrar uma confusão real e explicar cobertura versus acurácia entre previsões aceitas. |
| Inferência e API | Pesos float32, paridade Python/JavaScript, FastAPI | Fazer uma requisição, inspecionar a resposta e explicar validação de entrada e abstinência. |
| RAG | Busca BM25 no navegador e embeddings no backend local | Mostrar o procedimento recuperado, a resposta gerada e uma falha de fundamentação. |
| Agente com ferramentas | `src/supportops/agent.py` e traces reais | Mostrar que o LLM solicita ferramentas, que o código valida os argumentos e que o orçamento é limitado. |
| Engenharia e entrega | Testes de paridade, integridade e API; GitHub Actions e Pages | Fazer uma alteração, rodar os testes e explicar o que a CI verifica sem depender de LLM ao vivo. |

## Limites que devem ser ditos com clareza

- **Treinamento de LLM e deep learning:** o modelo treinado é um classificador clássico. Não foi realizado fine-tuning, LoRA ou treinamento de rede neural.
- **Cloud e MLOps:** o site público demonstra inferência no navegador. O backend Python não foi implantado como serviço cloud. Não há monitoramento de drift, registry de modelos ou orquestração de treinamento.
- **SQL e engenharia de dados:** esta implementação lê arquivos públicos. Não prova modelagem de banco, ETL em escala ou consultas SQL complexas.
- **Produção:** não há operação bancária, autenticação de clientes, validação com um banco ou teste de carga de um serviço público.
- **Avaliação de RAG/agentes:** são casos de desenvolvimento pequenos e inspecionáveis. Falta um conjunto independente, avaliação humana e medição de fidelidade, falsas aceitações e ataques diversos.
- **Português:** o classificador foi avaliado em inglês. A tradução local por LLM é uma etapa adicional sujeita a erro; ela não transforma o teste inglês em validação multilíngue.

## Como usar o projeto para se preparar

1. Começar pelo [guia de estudo](GUIA_DE_ESTUDO.md), reproduzir a inferência e explicar cada campo da resposta.
2. Escolher cinco erros do classificador e registrar uma análise própria, sem alterar o teste para melhorar os números.
3. Escrever perguntas novas para avaliar recuperação e fundamentação separadamente.
4. Comparar o pedido original, a tradução, a intenção e as chamadas do agente; localizar a etapa responsável por um erro.
5. Escolher a próxima extensão a partir de uma vaga concreta: servir o backend em cloud, integrar SQL ou comparar um encoder neural supervisionado. Avaliar custo e escopo antes de implementar.

Uma forma honesta de apresentar o trabalho, depois de compreender e reproduzir as etapas, é: “Projeto educacional de classificação de chamados com treinamento e avaliação reproduzíveis; RAG e agente local com ferramentas de leitura, testes e análise de limitações.” A participação concreta de Daniel deve ser descrita conforme o trabalho que ele conseguir explicar e executar.
