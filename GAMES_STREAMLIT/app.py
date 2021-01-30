import streamlit as st
from sqlalchemy import create_engine, text
import pandas as pd
import urllib
from surprise import KNNBasic
from surprise import Dataset
from surprise import Reader
from PIL import Image

#Trecho adicionado para realizar a conexão e resultado do banco de dados.

parametros = urllib.parse.quote_plus('DRIVER={ODBC Driver 17 for SQL Server};SERVER=127.0.0.1,11434;DATABASE=gogamedb;UID=sa;PWD=Crudo167')
conexao = create_engine("mssql+pyodbc:///?odbc_connect={}".format(parametros))


def runQuery(sql):
    result = conexao.connect().execution_options(isolation_level="AUTOCOMMIT").execute((text(sql)))
    return pd.DataFrame(result.fetchall(), columns=result.keys())

sql_users_games = 'Select * from tbl_users_games'


#Colocando o processo de recomendacao de usuarios - INI
df = runQuery(sql_users_games)
#df = pd.read_csv("tbl_users_games.csv")

gogame = df.drop(columns=['id', 'platform_id', 'network_id', 'status_id'])

#Alterar o n para o algoritmo utilizar mais dados da base
gogame_fr = gogame.sample(n=6000, random_state=42)

print(gogame_fr)

rating_dic = {'user' : gogame_fr['user_id'],
             'game' : gogame_fr['game_id'],
             'favorite': gogame_fr['is_favorite']}


def FriendRecommender (user):
    df = pd.DataFrame (rating_dic)
    reader = Reader(rating_scale=(0,1))
    data = Dataset.load_from_df(df[['user', 'game', 'favorite']], reader)
    trainset = data.build_full_trainset()
    sim_options = {'name' : 'cosine',
                  'user_based' : True}
    
    algo = KNNBasic(sim_options)
    algo.fit(trainset)
    
    uid = trainset.to_inner_uid(user) 
    pred = algo.get_neighbors(uid, 3)

    list_result = []
    
    for i in pred:
        #print(trainset.to_raw_uid(i))
        list_result.append(trainset.to_raw_uid(i))

    return list_result
#Colocando o processo de recomendacao de usuarios - FIM

#Carregando o titulo
st.markdown('# Projeto GO GAME')
# adicionando uma descrição
#st.write('Esse é um dashboard para o projeto Go Game, onde irá simular o processo de indicação de novas conexões e jogos.')
st.markdown('Essa é uma pagina para o projeto Go Game, onde o objetivo é gerar recomendações de conexões com outros usuários do App.')

st.write('')

#Carregando as imagens
image_controle = Image.open('images/controle_video_games.jpg')
st.image(image_controle, caption='source: https://unsplash.com/photos/jrqeb5o7H2U',use_column_width=True)

st.write('')

st.header('Recomendar novas conexões')

st.markdown('Para obter indicações de novas **conexões**, informe seu Id de usuário do app. :sign_of_the_horns:')

#Fazendo um input de dados
#id_user = st.number_input('Informe seu Id de usuário')
id_user = st.text_input('Informe seu Id de usuário')
if id_user:
    sql_user = f'''Select
    tg.name
From tbl_users tu
Join tbl_users_games tug
	on tu.id = tug.user_id
Join tbl_games tg
	on tug.game_id = tg.id
Where
	tu.id ={id_user}'''
    st.markdown('Esses são os jogos que você tem no App! :video_game:')
    #Realizando a consulta no banco
    resultado = runQuery(sql_user) 
    list_games = resultado.values.tolist()

    if(len(list_games) > 0):
        lista_jogos = '<ul>'
        i = 0
        while i < len(list_games):
            game = list_games[i]
        
            lista_jogos += f'<li>{game[0]}</li>'
            i+= 1
        lista_jogos += '</ul>'
        st.markdown(lista_jogos, unsafe_allow_html=True)
        st.markdown('***')
        st.markdown('### Recomendamos as seguintes conexões:')
        amigos = FriendRecommender(int(id_user))
        #criando tres colunas, para melhor apresentação
        col0, col1, col2 = st.beta_columns(3)
        with col0:
            st.header('Player_' + str(amigos[0]))
        with col1:
            st.header('Player_' + str(amigos[1]))
        with col2:
            st.header('Player_' + str(amigos[2]))

        print(amigos)

else:
    st.error('Informe um valor!')

