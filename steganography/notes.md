Importante:
    Checar os tipos direitinho, tipo de imagem de entrada, tipo de arquivo de entrada, tipo de encodificação. Checar se precisa passar a imagem pra uint8. Checar de funciona pra imagem em preto e branco.

    Fazer um requirements.txt

    Checar qual forma de transforma de int pra bin é mais rápida

    Falar no relatório do problema com opencv.imshow
    Que por isso mudei pra scikit image e matplotlib
    E tive que instalar tbm o pyqt5 pra utilizar GUI

    Checar se funciona para todos os tamanhos de imagem

    Verificar se a forma de abrir a imagem aumenta o contraste.
    Queremos a imagem exatamente como ela está.

    Lembrar de pedir pra o usuário escolher os planos de bits [0, 1 e/ou 2]

    Lembrar de dificultar na hora de codificar pra ficar dificil de decodificar caso alguém descubra

    Verificar a condição de parada (invertar uma descente), pode até usar contagem de bytes mas pode atrasar nosso esquema (apesar de que deve funcionar legal) [colocar o tamanho da mensagem no começo da imagem!!!]

    Primeiro faz trabalho e relatório que resolvem o problema, depois incrementa!

    Fazer diff.py code

    Como que passa um texto pra binário, qual o tamanho? pq int eu sei que gasta 1 byte NESSE CASO
    e um char de um texto gasta quando espaço, 1 BYTE tbm? ou mais?
    
Testes:
    Checar se a mensagem é valida, e não for, apresentar o problema (caracter inválido, tipo `)
    Checar se o tamanho da mensagem cabe na imagem, se exceder, cancelar e avisar ao usuário.
    Informar a quantidade de bytes de informação com base na imagem que o usuário escolheu.
    Informar a quantidade de bytes de informação que a mensagem do usuário tem.
    Colocar uma telinha de progresso [/... X%] ...