### Urgente:
1. Verificar se estamos alterando os planos de bits corretos. Ter certeza disso. OK.
2. Fazer código para mostrar plano de bits da image, e imagem... OK
3. FAZER RELATÓRIO
4. for idx in range(3): img[ : , : , idx].map(func) com função para transformar de int pra bin e vice versa.
5. Revisar requirements.txt

### Importante:
1. É melhor colocar pasta pra imagem e mensagem de saída mesmo. Ok? Depois modificar isso
2. Checar os tipos direitinho, tipo de imagem de entrada, tipo de arquivo de entrada.
3. Imagem PNG com transparência como fica? cv2.IMREAD_UNCHANGED?
4. Checar se funciona para todos os tamanhos de imagem
5. Verificar se a forma de abrir a imagem aumenta o contraste. Queremos a imagem exatamente como ela está.
6. Informar a porcentagem da imagem que foi modificada

### Testes:
1. Checar se a mensagem é valida, e se não for, apresentar o problema (caracter inválido, ...)
2. Checar se o tamanho da mensagem cabe na imagem, se exceder, cancelar e avisar ao usuário.
3. Testar imagem sem nenhum char.
4. Teste de velocidade de diferentes implementações e funções implementadas.
5. Testar erro de diretório da imagem.
6. Teste de consistencia de tamanho da mensagem.
7. Testes de checagem dos planos de bits

### Escolhas de implentação:
1. Tamanho da mensagem é a primeira coisa codificada dentro da imagem. 
2. Leitura da mensagem gasta mais memória mais acredito ser mais rápida... Ver isso direitinho

