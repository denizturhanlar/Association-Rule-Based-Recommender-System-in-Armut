# A description of the Apriori algorithm

![apriori](https://user-images.githubusercontent.com/126112467/230490297-93b2984c-b3c6-4d6e-9d98-6624899f6c6e.png)

The Apriori algorithm is a widely used data mining method that identifies frequent itemsets in a dataset. Its main objective is to discover patterns of items or itemsets that commonly appear together in the dataset. Initially, the algorithm determines the frequency of individual items in the dataset, and subsequently, it identifies combinations of these items, such as pairs, triples, and so on, until all combinations are determined. The process is iterated until no further combinations can be identified. Due to its ability to handle large datasets, the Apriori algorithm is employed in various fields, including data mining, market analysis, recommendation systems, and web page analysis.
Here are some examples of areas of use:
-Market analysis
-Recommendation systems


# Armut Product Recommendation System with Association Rule Learning

![armut](https://user-images.githubusercontent.com/126112467/230490145-993bf731-75de-4c09-9e49-1452cccd1c4b.png)

Armut, which is the biggest online service platform in Turkey, connects individuals who require services with service providers. With just a few clicks on a computer or smartphone, users can easily access a range of services, including cleaning, transportation, and renovation. The objective is to develop a product recommendation system utilizing Association Rule Learning, based on a dataset containing service users, the services they have received, and their corresponding categories.

# Dataset Story
The dataset comprises information about the services that customers receive and their respective categories, as well as the date and time at which each service is provided.

- UserId: Customer number 
- ServiceId: Anonymized services belonging to each category. 
- (Example: Upholstery washing service under Cleaning category) 
- A ServiceId can be found under different categories and represents different services under different categories. 
- (Example: While the service with CategoryId 7 and ServiceId 4 is honeycomb cleaning, the service with CategoryId 2 and ServiceId 4 is furniture assembly) 
- CategoryId: Anonymized categories. (Example: Cleaning, shipping, renovation category) 
- CreateDate: The date the service was purchased




