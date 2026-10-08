# Etapa 1: prompt usado com a IA generativa

O enunciado pede que uma IA generativa produza um `app.log` de **1 milhão de linhas** a partir de 5 linhas
de exemplo. Um chat não devolve 1 milhão de linhas (equivale a uns 60 MB de texto, muito acima do limite de saída
de qualquer modelo). Por isso a IA foi usada para escrever o **gerador**, e o arquivo sai da execução dele.
É a mesma ideia de geração de dados sintéticos do Módulo 3 (*Synthetic Data Generation*): dados artificiais
que preservam a distribuição e os padrões do mundo real.

IA utilizada: Claude (Anthropic), via Claude Code.

## Prompt

> Preciso de um arquivo de log chamado `app.log` com **exatamente 1.000.000 de linhas** para treinar e avaliar
> um analisador de logs com IA. Cada linha segue o formato `YYYY-MM-DD HH:MM:SS LEVEL message`, com LEVEL em
> INFO, DEBUG, WARN ou ERROR, como neste exemplo:
>
> ```
> 2024-11-08 10:00:00 INFO User admin logged in
> 2024-11-08 10:01:20 ERROR Failed to connect to database
> 2024-11-08 10:02:15 DEBUG Query executed in 120ms
> 2024-11-08 10:05:30 INFO User admin accessed dashboard
> 2024-11-08 10:10:00 WARN Connection pool running low
> ```
>
> Como um chat não comporta 1 milhão de linhas, escreva um **script Python (`gerar_logs.py`)** que gere o
> arquivo, com estes requisitos:
>
> 1. Período de 7 dias (2024-11-04 a 2024-11-10), linhas em ordem cronológica.
> 2. **Sazonalidade realista**: mais tráfego em horário comercial, madrugada calma, fim de semana com ~45% do
>    volume de um dia útil.
> 3. Vocabulário de uma aplicação web: login/logout de ~300 usuários, acesso a páginas, requisições de API com
>    latência, queries com tempo em ms (mediana ~35 ms, ~4% acima de 100 ms), cache, pool de conexões, jobs
>    agendados, exportação de relatórios, retries e timeouts de um gateway de pagamento.
> 4. Manter as mensagens do exemplo (`Failed to connect to database`, `Query executed in Nms`,
>    `Connection pool running low`...) para que o código guiado do enunciado também funcione.
> 5. **Ruído de fundo**: erros e falhas de login isolados ao longo da semana, para que "ter um ERROR" não
>    signifique "ter um incidente".
> 6. **Plantar 12 incidentes de 6 tipos** em horários variados (dia, noite, fim de semana), sobrepostos ao
>    tráfego normal: queda do banco, degradação de latência, força bruta de login vinda de IP externo,
>    ação administrativa de madrugada (permissões + exportação de tabela inteira), esgotamento do pool de
>    conexões e exportação em massa por um único usuário. Incidentes de segurança devem quase não gerar
>    ERROR, porque é assim que eles aparecem na vida real.
> 7. Gravar um **gabarito** `incidentes.json` com tipo, início e fim de cada incidente, para medir se o
>    analisador encontra o que deveria.
> 8. Seed fixa (reprodutível) e um modo `--live` que escreva linhas com o horário atual e injete incidentes
>    periodicamente, para demonstrar o `ai_service` em tempo real com Prometheus e Grafana.

## Resultado

- [ai_service/gerar_logs.py](../ai_service/gerar_logs.py): gera `data/app.log` (1.000.000 linhas, ~61 MB) e
  `data/incidentes.json` em ~10 s.
- Distribuição obtida: INFO 61,4% · DEBUG 30,8% · WARN 6,2% · ERROR 1,6%; 82 templates distintos de mensagem.
- Ajuste feito depois da primeira análise: no tráfego normal a conta `admin` agia a qualquer hora, o que
  tornava "admin de madrugada" um ruído comum. O gerador passou a usar `admin` só em horário comercial de
  dias úteis, como numa empresa real.
