import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import faiss
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

@app.route("/recommender/profile", methods=['POST'])
def get_collaborative_recommendations():
    # --- Carregar dados ---
    data = request.get_json()
    
    user_id = data.get("userId")
    
    reviews_df = pd.DataFrame(data.get('Review'))
    restaurants_df = pd.DataFrame(data.get('Restaurant'))
    
    # --- 1. Criar Matriz Usuário-Restaurante ---
    # Linhas = usuários, Colunas = restaurantes, Valores = notas
    user_item_matrix = reviews_df.pivot_table(
        index='userId',
        columns='restaurantId',
        values='rating',
        fill_value=0
    )
    print(user_item_matrix.head())
    
    # --- 2. Calcular Similaridade entre Usuários ---
    # Usa Similaridade de Cosseno
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )
    print(user_similarity_df.head());
    
    # --- 3. Encontrar Usuários Similares ---
    if user_id not in user_similarity_df.index:
        return {"error": "Usuário não encontrado ou sem avaliações"}, 404
    
    # Pega os 5 usuários mais similares (excluindo o próprio usuário)
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6]
    print(similar_users)
    
    # --- 4. Encontrar Restaurantes para Recomendar ---
    # Restaurantes já visitados pelo usuário
    user_visited = set(reviews_df[reviews_df['userId'] == user_id]['restaurantId'])
    
    # Coletar restaurantes bem avaliados (nota >= 4) pelos usuários similares
    recommendations_scores = {}
    
    for similar_user_id, similarity_score in similar_users.items():
        # Avaliações positivas do usuário similar
        similar_user_reviews = reviews_df[
            (reviews_df['userId'] == similar_user_id) & 
            (reviews_df['rating'] >= 4)
        ]
        
        for _, review in similar_user_reviews.iterrows():
            restaurant_id = review['restaurantId']
            
            # Ignora restaurantes já visitados
            if restaurant_id in user_visited:
                continue
            
            # Pontuação ponderada pela similaridade
            score = similarity_score * review['rating']
            
            if restaurant_id in recommendations_scores:
                recommendations_scores[restaurant_id] += score
            else:
                recommendations_scores[restaurant_id] = score
    
    # --- 5. Ordenar e Retornar Top Recomendações ---
    if not recommendations_scores:
        return {
            "message": "Nenhuma recomendação encontrada",
            "user": find_user_name(user_id, data),
            "restaurants": []
        }
    
    # Ordena por pontuação
    sorted_recommendations = sorted(
        recommendations_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]  # Top 10
    
    # Formata resposta
    recommended_restaurants = []
    for restaurant_id, score in sorted_recommendations:
        restaurant = restaurants_df[restaurants_df['restaurantId'] == restaurant_id].iloc[0]
        recommended_restaurants.append({
            "restaurantId": restaurant_id,
        })
    
    return jsonify({
        "restaurants": recommended_restaurants
    })


# Função auxiliar
def find_user_name(user_id, data):
    user = next((u for u in data.get("Users") if u["userId"] == user_id), None)
    return user["name"].upper() if user else "Unknown"

@app.route("/recommender/similar", methods=['POST'])
def get_content_recommendations():

    data = request.get_json()

    # --- Entrada do usuário ---
    reviews_df = pd.DataFrame(data.get("Review"))

    # Definir o usuário para a recomendação
    user_request_id = data.get("UserRequestId")

    # Restaurantes já avaliados
    restaurantes_ja_visitados = set(
        reviews_df[reviews_df['userId'] == user_request_id]['restaurantId']
    )

    # Restaurantes base (avaliados com nota >= 3)
    positive_reviews = reviews_df[
        (reviews_df['userId'] == user_request_id) & (reviews_df['rating'] >= 3)
    ]
    restaurantes_base_recommendation = positive_reviews['restaurantId'].tolist()

    # Restaurantes DataFrame
    restaurantes_df = pd.DataFrame(data.get("Restaurant"))

    # --- Pré-processamento ---
    # 1. OneHotEncoding da categoria
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    categorias_encoded = ohe.fit_transform(restaurantes_df[['restaurantType']])
    categoria_df = pd.DataFrame(categorias_encoded, columns=ohe.get_feature_names_out(['restaurantType']))

    # 2. Normalização de preço
    scaler = MinMaxScaler()
    precos_escalados = scaler.fit_transform(restaurantes_df[['averagePrice']])
    precos_df = pd.DataFrame(precos_escalados, columns=['preco_medio_escalado'])

    # 3. Vetores finais
    restaurante_features_df = pd.concat([categoria_df, precos_df], axis=1)
    restaurant_vectors = restaurante_features_df.to_numpy().astype('float32')

    # --- FAISS ---
    dimension = restaurant_vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(restaurant_vectors)

    # --- Recomendações ---
    recomendacoes_finais = set()
    k = 5
    for ref_id in restaurantes_base_recommendation:
        ref_index = restaurantes_df[restaurantes_df['restaurantId'] == ref_id].index[0]
        user_vector = restaurant_vectors[ref_index].reshape(1, -1)
        distances, indices = index.search(user_vector, k)
        for i in indices[0]:
            restaurante_id_recomendado = restaurantes_df.iloc[i]['restaurantId']
            recomendacoes_finais.add(restaurante_id_recomendado)

    # Remover já visitados
    recomendacoes_filtradas = [r for r in recomendacoes_finais if r not in restaurantes_ja_visitados]
    restaurantes_recomendados_df = restaurantes_df[restaurantes_df['restaurantId'].isin(recomendacoes_filtradas)]

    # --- Output ---
    recommended_restaurants = []

    print(f"\n--- RECOMENDAÇÃO DE RESTAURANTES PARA O USUÁRIO {user_request_id} ---")
    if restaurantes_recomendados_df.empty:
        print("Nenhuma nova recomendação encontrada.")
        return {"message": "Nenhuma nova recomendação encontrada.", "restaurants": []}
    else:
        for _, row in restaurantes_recomendados_df.iterrows():
            print(f"Nome: {row['name']} | Categoria: {row['restaurantType']} | Preço Médio: R${row['averagePrice']:.2f}")
            restaurant_data = {
                "name": row['name'],
                "categoria": row['restaurantType'],
                "preco_medio": float(f"{row['averagePrice']:.2f}")
            }
            
            recommended_restaurants.append(restaurant_data)

        return {
            "user": user_request_id, 
            "total_restaurants": len(recommended_restaurants),
            "restaurants": recommended_restaurants
        }
    
@app.route("/recommender/onboarding", methods=['POST'])
def get_onboarding_recommendations():
    data = request.get_json()
    
    # Preferências do usuário novo
    preferred_types = data.get("preferredTypes", [])
    max_price = data.get("maxPrice", 100)
    min_rating = 3
    
    restaurants_df = pd.DataFrame(data.get('Restaurant'))
    reviews_df = pd.DataFrame(data.get('Review'))
    
    # Calcular média de avaliações
    avg_ratings = reviews_df.groupby('restaurantId')['rating'].mean()
    
    # Filtrar por preferências
    filtered = restaurants_df[
        (restaurants_df['restaurantType'].isin(preferred_types)) &
        (restaurants_df['averagePrice'] <= max_price)
    ]
    
    # Adicionar rating médio
    filtered['avg_rating'] = filtered['restaurantId'].map(avg_ratings)
    
    # Ordenar por rating
    recommendations = filtered[
        filtered['avg_rating'] >= min_rating
    ].sort_values('avg_rating', ascending=False)
    
    # Retornar apenas restaurantId
    recommended_restaurants = [
        {"restaurantId": rest_id} 
        for rest_id in recommendations['restaurantId']
    ]
    
    return jsonify({
        "restaurants": recommended_restaurants
    })

if __name__ == '__main__':
    app.run(debug=True, port=8000)