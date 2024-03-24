### Importante:
1. É melhor colocar pasta pra imagem e mensagem de saída mesmo. Ok? Depois modificar isso
2. Checar os tipos direitinho, tipo de imagem de entrada, tipo de arquivo de entrada, tipo de encodificação. Checar se precisa passar a imagem pra uint8. Checar de funciona pra imagem em preto e branco.
3. Imagem PNG com transparência como fica? cv2.IMREAD_UNCHANGED?
4. Checar se o requirements.txt tá certinho
5. Checar qual forma de transforma de int pra bin é mais rápida
6. Falar no relatório do problema com opencv.imshow. Que por isso mudei pra scikit image e matplotlib. E tive que instalar tbm o pyqt5 pra utilizar GUI
7. Checar se funciona para todos os tamanhos de imagem
8. Verificar se a forma de abrir a imagem aumenta o contraste. Queremos a imagem exatamente como ela está.
9. Lembrar de pedir pra o usuário escolher os planos de bits [0, 1 e/ou 2]
10. Lembrar de dificultar na hora de codificar pra ficar dificil de decodificar caso alguém descubra
11. Fazer diff.py code

### Testes:
1. Checar se a mensagem é valida, e não for, apresentar o problema (caracter inválido, ...)
2. Checar se o tamanho da mensagem cabe na imagem, se exceder, cancelar e avisar ao usuário.
3. Informar a quantidade de bytes de informação com base na imagem que o usuário escolheu.
4. Informar a quantidade de bytes de informação que a mensagem do usuário tem.
5. Colocar uma telinha de progresso [/... X%] ...