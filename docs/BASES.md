# Fontes, classes e decisões

As fontes UCI foram consultadas antes dos experimentos. Os metadados e a tabela de variáveis são preservados em `data/raw/<base>/`. A tabela `results/datasets.csv` registra dimensões efetivas, ausências, constantes, duplicatas, grupos conflitantes, contagens por classe, identificadores excluídos e SHA-256 do CSV original preservado. As dimensões esperadas são verificadas com erro explícito em caso de divergência.

| Base e fonte UCI | Atributos | Classe 0 | Classe 1 |
|---|---|---|---|
| [Sonar, 151](https://archive.ics.uci.edu/dataset/151) | 60 energias por banda | R: rocha | M: mina/cilindro metálico |
| [Ionosphere, 52](https://archive.ics.uci.edu/dataset/52) | 34 descritores de retornos de radar | b: retorno ruim | g: retorno bom |
| [Banknote, 267](https://archive.ics.uci.edu/dataset/267) | 4 estatísticas wavelet | rótulo original 0 | rótulo original 1 |
| [Blood, 176](https://archive.ics.uci.edu/dataset/176) | recência, frequência, volume e tempo | não doou em março/2007 | doou em março/2007 |
| [Climate, 252](https://archive.ics.uci.edu/dataset/252) | 18 parâmetros de simulação | falha | sucesso |
| [QSAR, 254](https://archive.ics.uci.edu/dataset/254) | 41 descritores moleculares | NRB: não prontamente biodegradável | RB: prontamente biodegradável |
| [Raisin, 850](https://archive.ics.uci.edu/dataset/850) | 7 descritores morfológicos | Kecimen | Besni |
| [Rice, 545](https://archive.ics.uci.edu/dataset/545) | 7 descritores morfológicos | Cammeo | Osmancik |
| [Spambase, 94](https://archive.ics.uci.edu/dataset/94) | 57 frequências textuais e estatísticas de maiúsculas | não spam | spam |
| [Retinopathy, 329](https://archive.ics.uci.edu/dataset/329) | 19 descritores de imagens retinianas | ausência de sinais | presença de sinais |

Banknote contém imagens de espécimes genuínos e falsificados, mas a documentação UCI consultada não explicita qual número corresponde a cada significado. Preservamos 0/1 sem inventar essa correspondência. Precisão e sensibilidade referem-se ao rótulo 1; a limitação semântica permanece documentada.

## Aquisição e limpeza

`ucimlrepo` é o caminho principal. Climate e QSAR possuem fallback para seus ZIPs oficiais quando o endpoint não suporta importação. Não há substituição de bases. Climate usa o cabeçalho de `pop_failures.dat`, descartando as colunas de estudo e execução. QSAR usa a ordem dos 41 descritores documentada na fonte; os nomes operacionais `descriptor_1` a `descriptor_41` preservam essa ordem. O ZIP original também é salvo.

Os dados numéricos são convertidos com erro explícito para entradas não numéricas; infinitos impedem a execução. Rótulos desconhecidos ou ausentes também a impedem. Ausências numéricas são imputadas por mediana dentro de cada treino. Constantes observadas na auditoria não são removidas globalmente: cada pipeline aprende sua remoção no próprio treino. Não eliminar correlações entre atributos, outliers ou observações desbalanceadas.

## Duplicatas e dependência

Auditoria efetivamente obtida antes da execução (duplicatas contam linhas adicionais após a primeira ocorrência):

| Base | Amostras × atributos | Classe 0 / 1 | Duplicatas de X | Grupos com alvos conflitantes | Tratamento específico |
|---|---|---|---|---|---|
| Sonar | 208 × 60 | 97 / 111 | 0 | 0 | folds estratificados |
| Ionosphere | 351 × 34 | 126 / 225 | 1 | 0 | agrupar duplicatas; remover Attribute2 constante no treino |
| Banknote | 1372 × 4 | 762 / 610 | 24 | 0 | agrupar duplicatas |
| Blood | 748 × 4 | 570 / 178 | 246 | 31 | agrupar duplicatas; preservar conflitos |
| Climate | 540 × 18 | 46 / 494 | 0 | 0 | excluir Study e Run; folds estratificados |
| QSAR | 1055 × 41 | 699 / 356 | 3 | 0 | agrupar duplicatas |
| Raisin | 900 × 7 | 450 / 450 | 0 | 0 | folds estratificados |
| Rice | 3810 × 7 | 1630 / 2180 | 0 | 0 | folds estratificados |
| Spambase | 4601 × 57 | 2788 / 1813 | 394 | 3 | agrupar duplicatas; preservar conflitos |
| Retinopathy | 1151 × 19 | 540 / 611 | 5 | 0 | agrupar duplicatas |

Nenhuma célula ausente ou infinita foi encontrada nos arquivos obtidos. Embora a página UCI de Spambase indique ausências, o arquivo disponibilizado e carregado não apresentou células ausentes. Climate é fortemente desbalanceada a favor de sucesso; ali a especificidade mede a identificação de falhas (classe 0), e a sensibilidade mede sucesso (classe 1). Não inverter classes após observar desempenho.

Linhas com atributos exatamente iguais recebem o mesmo grupo, mesmo se seus alvos divergem; não há deduplicação nem correção de alvo. StratifiedGroupKFold mantém esses grupos juntos nos níveis externo e interno; a estratificação pode ser aproximada pelas restrições de grupos. Sem duplicatas, usa-se StratifiedKFold. São preservados índices originais e todas as atribuições de folds.

Climate contém três desenhos Latin hypercube de 180 execuções. O número da execução é um índice que reinicia por estudo, não um identificador de pessoa nem prova de medição repetida do mesmo vetor. Os estudos representam amostragens distintas dos parâmetros. O experimento estima desempenho em novas configurações dessa coleção, sem afirmar generalização para novos estudos; não se agrupa por apenas três estudos, o que inviabilizaria os cinco folds prescritos. Identificadores são excluídos e configurações idênticas, se encontradas, são agrupadas. A dependência de desenho permanece limitação.

Sonar contém sinais em diferentes ângulos de objetos; não há identificador suficiente para validar generalização entre objetos novos. Retinopathy não fornece identificador de paciente que permita verificar imagens repetidas da mesma pessoa. Spambase tem mensagens de fontes pessoais e nenhuma chave de remetente utilizável. Nas demais bases, não há chave de entidade repetida disponível além da igualdade observável de atributos. Portanto, não se pode garantir independência de entidades latentes; o isolamento comprovado cobre índices, duplicatas observáveis e todas as transformações aprendidas.

## Problemas de ambiente

O atalho `python` do Windows não era utilizável e o interpretador do pgAdmin não tinha `venv`. Após indicação do usuário, adotou-se exclusivamente o Conda `rec_pad`. A rede exige autorização fora do sandbox. O cache matplotlib foi direcionado ao projeto. O scikit-learn instalado avisa sobre depreciação de `SVC(probability=True)`; a configuração foi mantida para respeitar o protocolo, sem introduzir um calibrador diferente. Avisos de convergência e demais avisos são registrados por base, fold, etapa e modelo.
