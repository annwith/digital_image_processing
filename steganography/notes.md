### Urgente:
1. Verificar se estamos alterando os planos de bits corretos. Ter certeza disso. OK.
2. Fazer código para mostrar plano de bits da image, e imagem... OK
3. Fazer diff.py code
4. for idx in range(3): img[ : , : , idx].map(func) com função para transformar de int pra bin e vice versa.
5. Verificar possibilidade de vetorizar.
6. Revisar requirements.txt

### Importante:
1. É melhor colocar pasta pra imagem e mensagem de saída mesmo. Ok? Depois modificar isso
2. Checar os tipos direitinho, tipo de imagem de entrada, tipo de arquivo de entrada, tipo de encodificação. Checar se precisa passar a imagem pra uint8. Checar de funciona pra imagem em preto e branco.
3. Imagem PNG com transparência como fica? cv2.IMREAD_UNCHANGED?
4. Checar se o requirements.txt tá certinho
5. Falar no relatório do problema com opencv.imshow. Que por isso mudei pra scikit image e matplotlib. E tive que instalar tbm o pyqt5 pra utilizar GUI
6. Checar se funciona para todos os tamanhos de imagem
7. Verificar se a forma de abrir a imagem aumenta o contraste. Queremos a imagem exatamente como ela está.
8. Lembrar de pedir pra o usuário escolher os planos de bits [0, 1 e/ou 2]
9. Lembrar de dificultar na hora de codificar pra ficar dificil de decodificar caso alguém descubra
10. Informar a quantidade de bytes de informação com base na imagem que o usuário escolheu.
11. Informar a quantidade de bytes de informação que a mensagem do usuário tem.
12. Colocar uma telinha de progresso [/... X%] ...
13. Checar o tipo de entrada e saída especificado pelo prof e as imagens exemplo do site dele.
14. Ver o que dá pra vetorizar, mas manter a versão atual para comparação...

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

