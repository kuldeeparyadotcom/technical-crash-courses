## Overview

Azure Cosmos DB is a fully managed, globally distributed, multi-model NoSQL database service from Microsoft Azure, engineered for modern applications that demand low-latency, high availability, and elastic scalability. It functions as both a NoSQL and a vector database, capable of handling unstructured, semi-structured, structured, and vector data types.

### What it is

Azure Cosmos DB provides native support for various data models—document, graph, key-value, and columnar—exposed through multiple APIs, including NoSQL, MongoDB, Gremlin, Cassandra, and Table APIs. This flexibility allows developers to leverage familiar tools and frameworks. As a fully managed service, it automates database administration tasks like management, updates, patching, and capacity scaling, offering cost-effective serverless and automatic scaling options.

Key features include:

*   **Turnkey Global Distribution & Multi-Master Support:** Easily distributes data across multiple Azure regions with a few clicks and enables low-latency writes from any region, ensuring high availability and performance.
*   **Guaranteed Performance:** Offers single-digit millisecond latencies at the 99th percentile, 99.999% availability for reads and writes, and tunable consistency levels.
*   **Multi-Model & Multi-API:** Supports diverse data structures and allows migration of existing applications using various NoSQL APIs.
*   **Integrated AI Capabilities (as of 2025):** Recent updates include efficient vector indexing and search, full-text search, and seamless integration with Azure AI Services to support Retrieval Augmented Generation (RAG) scenarios. A Query Copilot (preview) helps generate NoSQL queries from natural language prompts.
*   **Analytical Store:** Provides a fully isolated column store for large-scale analytics on operational data without impacting transactional workloads, eliminating complex ETL pipelines.
*   **Change Feed & Time to Live (TTL):** Enables clients to subscribe to data changes and automatically deletes items after a specified duration.

### What Problem It Solves

Azure Cosmos DB addresses the challenges of building modern, globally distributed applications that require:

*   **Extreme Scale and Performance:** It allows applications to handle massive amounts of data and high throughput with guaranteed low-order-of-millisecond response times globally.
*   **High Availability and Resilience:** With multi-master replication and global distribution, it ensures applications remain online and data is accessible even during regional outages.
*   **Developer Agility:** Its multi-model and flexible schema capabilities enable rapid development and iteration for applications with evolving data requirements. It simplifies the data infrastructure by serving as a single database for operational data needs.
*   **Cost Efficiency:** Offers serverless and automatic scaling options that adjust capacity to match demand, optimizing costs.

### Alternatives

Competitors and alternatives to Azure Cosmos DB largely fall into cloud-based NoSQL and specialized database services:

*   **Cloud NoSQL Databases:** Amazon DynamoDB (often considered the best overall alternative), Google Cloud Firestore, MongoDB Atlas, Couchbase Server, Redis Enterprise, Aerospike.
*   **Graph Databases:** Amazon Neptune, Neo4j Graph Database, ArangoDB.
*   **Big Data NoSQL:** Google Cloud Bigtable (for large analytical and operational workloads, time-series data).
*   **Relational & Other Database Services** (considered in broader comparisons): Amazon RDS, Microsoft SQL Server, Oracle Database, PostgreSQL, Google Cloud SQL, Amazon Aurora.

### Primary Use Cases

Azure Cosmos DB is well-suited for a wide range of mission-critical applications that need to operate at global scale:

*   **IoT (Internet of Things) Applications:** Ideal for ingesting and processing high-velocity, real-time data streams from sensors and devices, enabling real-time analytics and decision-making for scenarios like smart cities.
*   **E-commerce Platforms:** Manages diverse data such as product catalogs, user profiles, and transaction histories, providing flexibility and scalability to handle peak shopping seasons and low-latency access for global users.
*   **Gaming Industry:** Supports dynamic player profiles, in-game achievements, and real-time interactions with low-latency access, crucial for a seamless gaming experience across distributed player bases.
*   **Web and Mobile Applications:** Powers social applications by storing chats, comments, and posts, and facilitates rich personalized user experiences by handling user-generated content and integrating with third-party services.
*   **AI and Machine Learning Applications:** Serves as a unified AI database with vector search capabilities, supporting Retrieval Augmented Generation (RAG) and other AI agent development requiring fast, operational data access.
*   **Financial Services:** Used for managing customer accounts, transactions, and financial data, meeting strict compliance requirements and ensuring low-latency access for a secure and efficient banking experience.
*   **SaaS Applications:** Provides the ability to store data across multiple regions, scale quickly, and ensure fast response times for multi-tenant applications.

## Technical Details

### Top 10 Key Concepts of Azure Cosmos DB

This section delves into the top 10 key concepts of Azure Cosmos DB, providing clear definitions, best practices, and common pitfalls for each.

#### 1. Request Units (RUs)

*   **Definition:** Request Units (RUs) are the performance currency in Azure Cosmos DB, abstracting the system resources (CPU, memory, IOPS) required to perform database operations. Every operation—be it a read, write, update, delete, or query—consumes a certain number of RUs. The amount of RUs consumed depends on factors like item size, indexing, number of properties, and query complexity.
*   **Best Practices:**
    *   **Optimize Queries:** Design queries to be as efficient as possible, leveraging partition keys and indexes to reduce RU consumption.
    *   **Right-size Throughput:** Use the Azure Cosmos DB capacity calculator for initial estimates and monitor RU consumption metrics (like normalized RU consumption) to fine-tune your provisioned throughput.
    *   **Batch Operations:** Use the bulk executor library or SDKs for efficient batch operations to save RUs.
    *   **Indexing Policy:** Customize indexing policies to only index properties relevant to your queries to reduce write RU costs and storage.
*   **Common Pitfalls:**
    *   **Over-provisioning/Under-provisioning:** Setting RUs too high leads to unnecessary costs, while setting them too low causes throttling (HTTP 429 errors), impacting application performance.
    *   **Ignoring RU Consumption:** Not monitoring the RU charge per operation, especially for complex queries or large writes, can lead to unexpected costs and performance issues.
    *   **Cross-Partition Queries:** Queries that scan multiple logical partitions are significantly more expensive in RUs.

#### 2. Global Distribution & Multi-Master Support

*   **Definition:** Azure Cosmos DB allows you to distribute your data globally across any number of Azure regions with a few clicks. With multi-master (multi-region writes) enabled, applications can perform writes in any configured region, and Cosmos DB automatically handles data replication and conflict resolution across these regions, ensuring low-latency writes and high availability.
*   **Best Practices:**
    *   **Geographically Proximity:** Deploy your application instances in the same Azure regions as your Cosmos DB replicas to minimize latency.
    *   **Multi-Region Writes:** Enable multi-region writes for applications requiring low-latency writes globally and maximum availability.
    *   **Conflict Resolution:** Understand and configure conflict resolution policies (e.g., Last Write Wins or custom logic) when using multi-master, especially if concurrent updates to the same item from different regions are possible.
*   **Common Pitfalls:**
    *   **Cross-Region Reads/Writes:** Directing read/write traffic to a remote region when a local replica is available will incur higher latency and RU costs.
    *   **Lack of Conflict Strategy:** Not planning for write conflicts in a multi-master setup can lead to data inconsistencies if not properly handled by a conflict resolution policy.
    *   **Strong Consistency with Multi-Master:** Strong consistency is not supported with multiple write regions because it would increase write latencies due to the need for synchronous replication and commitment across all regions.

#### 3. Consistency Models

*   **Definition:** Azure Cosmos DB offers five well-defined, SLA-backed consistency models: Strong, Bounded Staleness, Session, Consistent Prefix, and Eventual. These models provide a spectrum of tradeoffs between consistency (data freshness), availability, and latency, allowing developers to choose the right balance for their application needs.
*   **Best Practices:**
    *   **Session Consistency (Default):** This is the most common choice, providing a good balance for user-centric applications where a user typically sees their own writes immediately within a session, while other users eventually see the updates.
    *   **Bounded Staleness:** Use when you need global freshness with a tolerable delay, defined by a time interval or a number of versions.
    *   **Strong Consistency:** Reserve for scenarios where absolute data precision is critical (e.g., financial transactions) and you can tolerate higher write latencies and reduced availability during region failures.
    *   **Evaluate Trade-offs:** Understand how each consistency level impacts read/write latency, throughput, and data durability for your specific workload.
*   **Common Pitfalls:**
    *   **Over-reliance on Strong Consistency:** Choosing strong consistency without understanding its performance implications (higher latency, lower throughput) can negatively impact application responsiveness, especially in multi-region setups.
    *   **Misunderstanding Eventual Consistency:** Assuming "eventual" means "instantly" or not accounting for potential data staleness in certain read operations can lead to application logic errors.
    *   **Ignoring Consistency per Request:** While a default is set, consistency can be overridden per request, which can lead to unexpected behavior if not managed carefully.

#### 4. Partitioning (Logical and Physical)

*   **Definition:** Partitioning is fundamental to Azure Cosmos DB's scalability, distributing data across multiple logical and physical partitions. A **logical partition** is a set of items that share the same partition key value (e.g., all documents for `userId: "Alice"`). A **physical partition** is an internal unit of scale, comprising one or more logical partitions, with its own allocated compute and storage. Cosmos DB automatically manages physical partitions as data and throughput scale.
*   **Best Practices:**
    *   **High Cardinality Partition Key:** Choose a partition key with a wide range of unique values (high cardinality) to ensure even distribution of data and requests across logical partitions. User ID, product ID, or a composite key are often good choices.
    *   **Distribute Reads and Writes Evenly:** The goal is to avoid "hot partitions" where a single logical partition receives a disproportionately high volume of requests, leading to throttling.
    *   **Co-locate Related Data:** Group frequently accessed data together in the same logical partition to enable efficient point reads and queries within a single partition, reducing RU costs.
    *   **Immutable Partition Key:** The partition key cannot be changed after an item is created, so careful planning is essential.
*   **Common Pitfalls:**
    *   **Low Cardinality Partition Key:** Choosing a key with limited unique values (e.g., `city`, `date` for a global app) can lead to hot partitions, performance bottlenecks, and increased costs.
    *   **Frequent Cross-Partition Queries:** Queries that don't include the partition key in their filter clause result in "fan-out" queries that scan all physical partitions, which are expensive and inefficient.
    *   **Large Logical Partitions:** While a logical partition can grow large, a single logical partition has a throughput limit (e.g., 10,000 RUs/sec for a physical partition it resides on), which can become a bottleneck even if data is evenly distributed otherwise.

#### 5. Multi-Model & Multi-API

*   **Definition:** Azure Cosmos DB is a multi-model database, natively supporting document, graph, key-value, and columnar data models. It exposes these models through various APIs, including NoSQL (its native API), MongoDB, Apache Cassandra, Apache Gremlin, and Azure Table. This allows developers to use familiar tools, SDKs, and query languages.
*   **Best Practices:**
    *   **Choose API Based on Workload/Existing Apps:** Select the API that best fits your application's data model and leverages existing team skills or application code. For new applications, the NoSQL API often provides the most direct access to Cosmos DB's native features.
    *   **Leverage API-Specific Features:** Understand that while core Cosmos DB features (global distribution, scaling) are shared, each API has specific capabilities and limitations that might influence data modeling and query patterns.
    *   **Once Chosen, Cannot Change:** Be aware that once an API is chosen for an account, it cannot be changed.
*   **Common Pitfalls:**
    *   **Treating APIs as Identical:** Assuming all APIs behave identically or offer the exact same feature set as their open-source counterparts can lead to unexpected behavior or missing functionality.
    *   **Using the Wrong API:** Choosing an API that doesn't align with the data model or access patterns (e.g., using the Table API for complex document relationships) can lead to inefficient solutions.
    *   **Rewriting Data Access:** If migrating an existing application, the primary goal of the multi-API support is to avoid rewriting the entire data access layer, so choose the API that aligns with your current system.

#### 6. Data Modeling

*   **Definition:** Data modeling in Azure Cosmos DB involves structuring your data within JSON documents (for NoSQL API) or other appropriate formats (for other APIs) to optimize for performance, scalability, and cost. Unlike relational databases, it typically involves denormalization and embedding to keep related data together and minimize expensive cross-partition queries and joins (which are not supported across containers).
*   **Best Practices:**
    *   **Optimize for Access Patterns:** Design your data model based on how your application will query and access data, prioritizing frequently executed queries.
    *   **Denormalize and Embed:** Embed related entities within a single document when they are frequently accessed together and the embedded data doesn't grow unboundedly. This reduces the need for multiple round trips to the database.
    *   **Reference When Necessary:** For large, infrequently accessed, or frequently changing related data, use referencing (similar to foreign keys) and handle joins in your application logic.
    *   **Item Size:** Keep item sizes under the 2MB limit (for NoSQL API) and consider that larger documents consume more RUs for operations.
*   **Common Pitfalls:**
    *   **Relational Mindset:** Applying traditional relational database normalization principles (e.g., creating many separate containers for different entity types) leads to inefficient cross-container lookups and costly queries, as Cosmos DB doesn't support joins across containers.
    *   **Unbounded Arrays/Embedded Data:** Embedding lists that can grow infinitely large can lead to large documents, exceeding the 2MB item limit and increasing RU costs.
    *   **Ignoring Partition Key in Modeling:** Not considering the partition key early in data modeling can lead to poor data distribution and performance issues.

#### 7. Change Feed

*   **Definition:** The Azure Cosmos DB change feed provides a persistent, ordered, and real-time record of all changes (inserts, updates, and deletes) to items within a container. It allows applications to listen for these changes and react to them, enabling various real-time processing scenarios without complex ETL pipelines.
*   **Best Practices:**
    *   **Change Feed Processor Library:** For most scenarios, use the Change Feed Processor library (available in SDKs) or Azure Functions triggers, as they handle leasing, checkpointing, and distributing processing across multiple consumers automatically.
    *   **Idempotent Consumers:** Design your consumers to be idempotent, meaning processing the same change multiple times produces the same result, as the change feed might deliver changes more than once in certain failure scenarios.
    *   **Soft Deletes:** Since the change feed does not directly contain deleted documents (only their current state), implement a soft-delete mechanism (e.g., an `IsDeleted` flag) if you need to capture deletions through the change feed.
    *   **Use Cases:** Ideal for real-time stream processing, event sourcing, data synchronization (cache, search index, data warehouse), and triggering notifications.
*   **Common Pitfalls:**
    *   **Handling Deletes Incorrectly:** Expecting deleted documents to appear in the change feed (as opposed to their final state or relying on a soft-delete flag) can lead to data inconsistencies in downstream systems.
    *   **Manual Polling:** Implementing custom polling logic instead of using the Change Feed Processor can be complex to manage for scale and reliability.
    *   **Not Considering Order:** While order is guaranteed within a partition key, it's not guaranteed across partition keys. Design your processing logic accordingly if cross-partition order is critical.

#### 8. Analytical Store (Azure Synapse Link)

*   **Definition:** The Analytical Store is a fully isolated, column-oriented store that automatically syncs operational data from an Azure Cosmos DB container. It enables large-scale analytics and ETL-free data processing using Azure Synapse Analytics without impacting the performance of transactional workloads in the operational (row-based) store.
*   **Best Practices:**
    *   **Enable for Large Analytics:** Use Analytical Store when you need to run complex analytical queries or generate reports on your operational data without consuming provisioned throughput from the transactional store.
    *   **Integrate with Azure Synapse:** Leverage Azure Synapse Spark or Synapse SQL Serverless pools to query the Analytical Store, as this is the primary way to access it.
    *   **Schema Flexibility:** Benefit from the Analytical Store's schema flexibility, which automatically infers the schema from your operational data.
*   **Common Pitfalls:**
    *   **Using for OLTP:** Analytical Store is designed for OLAP (Online Analytical Processing) workloads, not real-time transactional (OLTP) operations.
    *   **Complex Schemas:** While flexible, very complex or highly nested schemas might require careful handling when queried from Synapse, especially if fields have whitespace or non-standard characters.
    *   **Not Understanding Data Types:** Certain BSON data types are not supported and won't be represented in the analytical store.

#### 9. Serverless & Autoscale Provisioned Throughput

*   **Definition:** Azure Cosmos DB offers flexible throughput provisioning options:
    *   **Serverless:** Ideal for unpredictable or intermittent workloads, you only pay for the RUs consumed and storage. There's no minimum RU charge.
    *   **Autoscale Provisioned Throughput:** Allows you to set a maximum RU/s, and Cosmos DB automatically scales throughput between 10% of the max RU/s and the max RU/s based on demand, reducing the need for manual scaling and optimizing costs for variable workloads.
*   **Best Practices:**
    *   **Serverless for Dev/Test & Low/Variable Workloads:** For new projects, infrequent workloads, or development/testing environments, serverless offers cost-effectiveness as you only pay for actual usage.
    *   **Autoscale for Variable Production Workloads:** Use autoscale for production applications with fluctuating traffic patterns to automatically handle spikes and troughs, ensuring performance while optimizing costs.
    *   **Manual Provisioned Throughput for Predictable Workloads:** If your workload has consistently steady and predictable traffic, manual provisioned throughput can be more cost-effective.
    *   **Capacity Calculator:** Use the Azure Cosmos DB capacity calculator to estimate throughput requirements for informed decisions between options.
*   **Common Pitfalls:**
    *   **Autoscale Billing Misunderstanding:** While autoscale is convenient, its billing rate per RU/s is typically higher (e.g., 1.5x) than standard manual throughput. For consistently low usage, manual provisioning might be cheaper if the minimum for autoscale is often hit.
    *   **Serverless for High Throughput:** Serverless has throughput limits per container. For consistently high-throughput workloads, provisioned throughput (manual or autoscale) is more suitable.
    *   **Ignoring Cost Implications:** Not understanding the cost implications of each model for your specific workload can lead to unexpected bills.

#### 10. Vector Database Capabilities & AI Integration

*   **Definition:** Azure Cosmos DB has evolved into a vector database, natively supporting efficient vector indexing and search. This capability allows it to store vector embeddings (numerical representations of unstructured data like text, images, or audio) alongside operational data. This is crucial for AI and Machine Learning applications, particularly for Retrieval Augmented Generation (RAG) scenarios, where LLMs can query custom, domain-specific data to generate more accurate responses.
*   **Best Practices:**
    *   **Unified AI Database:** Leverage Cosmos DB to store both your operational data and vector embeddings in one place, simplifying your architecture for RAG and other AI agent development.
    *   **Real-time RAG:** Combine Cosmos DB's low-latency and real-time ingestion capabilities with vector search for up-to-date and contextually relevant AI responses.
    *   **Integrate with Azure AI Services:** Seamlessly connect with Azure OpenAI and other Azure AI services for embedding generation and LLM interactions.
    *   **Leverage SDKs and Frameworks:** Utilize integrations with popular AI frameworks like Semantic Kernel, LangChain, and LlamaIndex for easier development of RAG applications.
*   **Common Pitfalls:**
    *   **Ignoring Latency Requirements:** While Cosmos DB is fast, the overall RAG latency also depends on embedding generation and LLM response times.
    *   **Suboptimal Embedding Models:** The quality of vector search is directly tied to the quality of the embeddings. Using an unsuitable or poorly trained embedding model can yield irrelevant search results.
    *   **Mismanaging Indexing:** Not properly configuring vector indexes (e.g., dimensions, distance function) can lead to inefficient searches.
    *   **Over-reliance on Vector Search:** Vector search is a powerful tool, but it's often best combined with traditional filtering (e.g., by partition key or other properties) to narrow down the search space and improve relevance.

### Real-World Code Snippets and Examples

Here are five top-notch, real-world code snippets and examples illustrating key aspects and the latest capabilities of Azure Cosmos DB in various programming languages:

#### 1. C# - Vector Database Capabilities & AI Integration (NoSQL API)

This example demonstrates how to store vector embeddings and perform similarity searches using Azure Cosmos DB's native vector database capabilities, crucial for Retrieval Augmented Generation (RAG) scenarios in AI applications. This utilizes the latest features enabling direct vector indexing and search.

**Scenario:** Storing text data along with its vector embeddings and performing a k-nearest neighbor (KNN) search to find similar documents.

**Prerequisites:**

*   Azure Cosmos DB account (NoSQL API) with vector search enabled.
*   .NET SDK (latest version).
*   An embedding model (e.g., Azure OpenAI's `text-embedding-ada-002`) to generate vector embeddings.

```csharp
using Microsoft.Azure.Cosmos;
using Microsoft.Azure.Cosmos.Linq;
using System;
using System.Linq;
using System.Threading.Tasks;

public class CosmosVectorSearch
{
    private static readonly string EndpointUri = "YOUR_COSMOS_DB_ENDPOINT_URI";
    private static readonly string PrimaryKey = "YOUR_COSMOS_DB_PRIMARY_KEY";
    private static readonly string DatabaseId = "VectorDb";
    private static readonly string ContainerId = "Documents";

    public static async Task RunVectorSearchExample()
    {
        using CosmosClient client = new CosmosClient(EndpointUri, PrimaryKey);
        Database database = await client.CreateDatabaseIfNotExistsAsync(DatabaseId);

        // 1. Configure Vector Index on the container (typically done once via Azure Portal or SDK)
        // For SDK, this involves setting up ContainerProperties with VectorEmbeddingPolicy.
        // Example conceptual setup for container creation/update:
        // This part would ideally be done during resource provisioning or once via management plane.
        // The actual `VectorEmbeddingPolicy` is part of the latest SDK versions.
        /*
        ContainerProperties containerProperties = new ContainerProperties(ContainerId, partitionKeyPath: "/id")
        {
            VectorEmbeddingsPolicy = new VectorEmbeddingPolicy
            {
                VectorEmbeddings =
                {
                    new VectorEmbedding { Path = "/vector", DataType = VectorDataType.Float32, Dimensions = 1536, DistanceFunction = VectorDistanceFunction.Cosine }
                }
            }
        };
        await database.CreateContainerIfNotExistsAsync(containerProperties);
        */
        Container container = await database.CreateContainerIfNotExistsAsync(ContainerId, "/id");


        // 2. Ingest data and create embeddings (typically via Azure OpenAI)
        // For demonstration, we'll use placeholder embeddings.
        // In a real application, you'd call an embedding service here.
        var doc1 = new
        {
            id = Guid.NewGuid().ToString(),
            text = "Azure Cosmos DB is a globally distributed database service for modern applications.",
            vector = new float[] { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f /* ... 1536 embedding values */ }
        };
        var doc2 = new
        {
            id = Guid.NewGuid().ToString(),
            text = "Vector search in Cosmos DB allows finding similar data points.",
            vector = new float[] { 0.11f, 0.21f, 0.31f, 0.41f, 0.51f /* ... 1536 embedding values */ }
        };
        var doc3 = new
        {
            id = Guid.NewGuid().ToString(),
            text = "Microsoft Azure offers a wide range of cloud services.",
            vector = new float[] { 0.05f, 0.15f, 0.25f, 0.35f, 0.45f /* ... 1536 embedding values */ }
        };
        var doc4 = new
        {
            id = Guid.NewGuid().ToString(),
            text = "Machine learning models often use vector embeddings.",
            vector = new float[] { 0.12f, 0.22f, 0.32f, 0.42f, 0.52f /* ... 1536 embedding values */ }
        };

        await container.CreateItemAsync(doc1, new PartitionKey(doc1.id));
        await container.CreateItemAsync(doc2, new PartitionKey(doc2.id));
        await container.CreateItemAsync(doc3, new PartitionKey(doc3.id));
        await container.CreateItemAsync(doc4, new PartitionKey(doc4.id));

        Console.WriteLine("Documents ingested with vector embeddings.");

        // 3. Perform Vector Similarity Search
        // Example query embedding (would come from your query text via an embedding model)
        var queryEmbedding = new float[] { 0.10f, 0.20f, 0.30f, 0.40f, 0.50f /* ... query embedding values */ };

        QueryDefinition query = new QueryDefinition(
            "SELECT TOP 3 c.id, c.text, VectorDistance(c.vector, @queryEmbedding, true) AS SimilarityScore " +
            "FROM c ORDER BY VectorDistance(c.vector, @queryEmbedding, true) ASC" // ASC for Cosine similarity (smaller distance = more similar)
        )
        .WithParameter("@queryEmbedding", queryEmbedding);

        using FeedIterator<dynamic> feedIterator = container.GetItemQueryIterator<dynamic>(query);

        Console.WriteLine("\nVector Similarity Search Results:");
        while (feedIterator.HasMoreResults)
        {
            FeedResponse<dynamic> response = await feedIterator.ReadNextAsync();
            foreach (var result in response)
            {
                Console.WriteLine($"ID: {result.id}, Text: {result.text}, Similarity: {result.SimilarityScore:F4}");
            }
        }
    }
}
```

#### 2. Python - Basic CRUD Operations (NoSQL API)

This example demonstrates fundamental Create, Read, Update, and Delete (CRUD) operations for document data using the Python SDK for Azure Cosmos DB's NoSQL API. It highlights handling partition keys and item operations.

**Scenario:** Managing a simple collection of "Product" items.

**Prerequisites:**

*   Azure Cosmos DB account (NoSQL API).
*   Python SDK (`azure-cosmos`).

```python
import asyncio
from azure.cosmos.aio import CosmosClient
from azure.cosmos import PartitionKey

# Replace with your Cosmos DB connection details
ENDPOINT = "YOUR_COSMOS_DB_ENDPOINT_URI"
KEY = "YOUR_COSMOS_DB_PRIMARY_KEY"
DATABASE_NAME = "ProductDb"
CONTAINER_NAME = "Products"

async def run_crud_example():
    client = CosmosClient(ENDPOINT, KEY)
    try:
        database = await client.create_database_if_not_exists(id=DATABASE_NAME)
        # Partition key is crucial for performance and scalability
        container = await database.create_container_if_not_exists(
            id=CONTAINER_NAME,
            partition_key=PartitionKey(path="/category"),
            offer_throughput=400 # Or use autoscale_max_throughput=4000
        )
        print(f"Database '{DATABASE_NAME}' and Container '{CONTAINER_NAME}' ensured.")

        # 1. Create an item
        product_1 = {
            "id": "item1",
            "name": "Laptop",
            "category": "Electronics",
            "price": 1200.00,
            "quantity": 50
        }
        created_item = await container.upsert_item(product_1)
        print(f"\nCreated/Upserted item: {created_item['id']} (RU: {container.client_connection.last_response_headers['x-ms-request-charge']})")

        product_2 = {
            "id": "item2",
            "name": "Mouse",
            "category": "Electronics",
            "price": 25.00,
            "quantity": 200
        }
        await container.upsert_item(product_2)

        # 2. Read an item (point read - most efficient)
        read_item = await container.read_item(item="item1", partition_key="Electronics")
        print(f"\nRead item: {read_item['id']}, Name: {read_item['name']}, Price: {read_item['price']} (RU: {container.client_connection.last_response_headers['x-ms-request-charge']})")

        # 3. Query items (with partition key for efficiency)
        query = "SELECT * FROM c WHERE c.category = @category AND c.price > @price"
        params = [
            {"name": "@category", "value": "Electronics"},
            {"name": "@price", "value": 100}
        ]
        print("\nQuery results:")
        items = container.query_items(query=query, parameters=params, enable_cross_partition_query=False)
        async for item in items:
            print(f"  - {item['id']}: {item['name']} - ${item['price']}")
        print(f"(RU for query: {container.client_connection.last_response_headers['x-ms-request-charge']})")

        # 4. Update an item
        read_item['price'] = 1150.00
        updated_item = await container.replace_item(item=read_item['id'], body=read_item, partition_key="Electronics")
        print(f"\nUpdated item: {updated_item['id']}, New Price: {updated_item['price']} (RU: {container.client_connection.last_response_headers['x-ms-request-charge']})")

        # 5. Delete an item
        await container.delete_item(item="item2", partition_key="Electronics")
        print(f"\nDeleted item 'item2' (RU: {container.client_connection.last_response_headers['x-ms-request-charge']})")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(run_crud_example())
```

#### 3. Java - Change Feed Processor (NoSQL API)

This Java example demonstrates how to use the Change Feed Processor library to react to data changes in real-time. This is a robust way to build event-driven architectures, integrate with other services, or update materialized views.

**Scenario:** Monitoring a "SensorReadings" container and processing new or updated sensor data as it arrives.

**Prerequisites:**

*   Azure Cosmos DB account (NoSQL API).
*   Java SDK (latest version).
*   A dedicated "leases" container in the same database for the Change Feed Processor to manage state.

```java
import com.azure.cosmos.*;
import com.azure.cosmos.models.*;
import com.azure.cosmos.util.CosmosPagedFlux;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;

public class ChangeFeedProcessorExample {

    private static final String ENDPOINT = "YOUR_COSMOS_DB_ENDPOINT_URI";
    private static final String KEY = "YOUR_COSMOS_DB_PRIMARY_KEY";
    private static final String DATABASE_NAME = "TelemetryDb";
    private static final String CONTAINER_NAME = "SensorReadings";
    private static final String LEASE_CONTAINER_NAME = "leases";

    private static CosmosAsyncClient client;
    private static CosmosAsyncDatabase database;
    private static CosmosAsyncContainer feedContainer;
    private static CosmosAsyncContainer leaseContainer;
    private static com.azure.cosmos.ChangeFeedProcessor changeFeedProcessor;

    public static void main(String[] args) throws InterruptedException {
        client = new CosmosClientBuilder()
                .endpoint(ENDPOINT)
                .key(KEY)
                .contentResponseOnWriteEnabled(false) // For performance
                .buildAsyncClient();

        database = client.createDatabaseIfNotExists(DATABASE_NAME).block().getDatabase();
        feedContainer = database.createContainerIfNotExists(CONTAINER_NAME, "/sensorId", new ThroughputProperties(400))
                .block().getContainer();
        leaseContainer = database.createContainerIfNotExists(LEASE_CONTAINER_NAME, "/id", new ThroughputProperties(400))
                .block().getContainer();

        System.out.println("Database and containers ensured.");

        // Start the Change Feed Processor
        startChangeFeedProcessor();

        // Simulate data ingestion
        System.out.println("\nSimulating data ingestion...");
        for (int i = 0; i < 5; i++) {
            JsonNode sensorReading = createSensorReading("sensor" + (i % 2), 20 + i);
            feedContainer.createItem(sensorReading).block();
            System.out.println("  Ingested: " + sensorReading.get("id").asText());
            Thread.sleep(1000);
        }

        System.out.println("\nWaiting for Change Feed Processor to process changes...");
        // Keep main thread alive to allow CFP to run
        Thread.sleep(Duration.ofSeconds(15).toMillis());

        stopChangeFeedProcessor();
        client.close();
        System.out.println("Change Feed Processor example finished.");
    }

    private static JsonNode createSensorReading(String sensorId, double temperature) {
        ObjectMapper mapper = new ObjectMapper();
        return mapper.createObjectNode()
                .put("id", UUID.randomUUID().toString())
                .put("sensorId", sensorId)
                .put("temperature", temperature)
                .put("timestamp", System.currentTimeMillis());
    }

    private static void startChangeFeedProcessor() {
        // The processor name should be unique for each instance of your application.
        // It helps to distribute leases correctly.
        String processorName = "SensorProcessor-" + UUID.randomUUID();

        changeFeedProcessor = new ChangeFeedProcessorBuilder()
                .hostName(processorName)
                .feedContainer(feedContainer)
                .leaseContainer(leaseContainer)
                .handleChanges((List<JsonNode> docs) -> {
                    System.out.println("\n--- Change Feed Batch Started ---");
                    for (JsonNode document : docs) {
                        System.out.println(String.format("  Processing change: ID = %s, SensorID = %s, Temperature = %.2f",
                                document.get("id").asText(),
                                document.get("sensorId").asText(),
                                document.get("temperature").asDouble()));
                        // Here you would add your business logic: send to another service, update a cache, etc.
                    }
                    System.out.println("--- Change Feed Batch Finished ---");
                })
                // Start from now for new items, or .fromBeginning() for all items
                .startFromBeginning()
                // Or .startFromFeedContinuation() if you have a continuation token
                .build();

        changeFeedProcessor.start()
                .doOnSuccess(aVoid -> System.out.println("Change Feed Processor started successfully."))
                .subscribe(); // Don't block, subscribe to keep it running
    }

    private static void stopChangeFeedProcessor() {
        if (changeFeedProcessor != null) {
            changeFeedProcessor.stop()
                    .doOnSuccess(aVoid -> System.out.println("Change Feed Processor stopped."))
                    .block(); // Block here to ensure stop completes
        }
    }
}
```

#### 4. JavaScript/Node.js - Efficient Partitioned Querying (NoSQL API)

This Node.js example demonstrates how to efficiently query data in a partitioned container, emphasizing the importance of including the partition key in queries to avoid expensive cross-partition requests.

**Scenario:** Querying user data where user activity is partitioned by `userId`.

**Prerequisites:**

*   Azure Cosmos DB account (NoSQL API).
*   Node.js SDK (`@azure/cosmos`).

```javascript
const { CosmosClient } = require("@azure/cosmos");

// Replace with your Cosmos DB connection details
const endpoint = "YOUR_COSMOS_DB_ENDPOINT_URI";
const key = "YOUR_COSMOS_DB_PRIMARY_KEY";
const databaseId = "UserActivityDb";
const containerId = "UserActivities";
const partitionKeyPath = "/userId";

async function runPartitionedQueryExample() {
    const client = new CosmosClient({ endpoint, key });
    const { database } = await client.databases.createIfNotExists({ id: databaseId });
    const { container } = await database.containers.createIfNotExists({
        id: containerId,
        partitionKey: { paths: [partitionKeyPath] },
        // Optionally set throughput here, e.g., throughput: 400
    });
    console.log(`Database '${databaseId}' and Container '${containerId}' ensured.`);

    // 1. Create sample data
    const user1Activity1 = {
        id: "activity1-user1",
        userId: "user1",
        type: "login",
        timestamp: Date.now() - 3600000, // 1 hour ago
        details: { ip: "192.168.1.1" }
    };
    const user1Activity2 = {
        id: "activity2-user1",
        userId: "user1",
        type: "purchase",
        timestamp: Date.now() - 1800000, // 30 mins ago
        details: { productId: "P123", amount: 99.99 }
    };
    const user2Activity1 = {
        id: "activity1-user2",
        userId: "user2",
        type: "view_product",
        timestamp: Date.now() - 7200000, // 2 hours ago
        details: { productId: "P456" }
    };

    await container.items.upsert(user1Activity1);
    await container.items.upsert(user1Activity2);
    await container.items.upsert(user2Activity1);
    console("\nSample activities ingested.");

    // 2. Efficient Query (includes partition key)
    const query1 = `SELECT * FROM c WHERE c.userId = @userId AND c.type = "purchase"`;
    const { resources: user1Purchases, requestCharge: charge1 } = await container.items.query({
        query: query1,
        parameters: [{ name: "@userId", value: "user1" }]
    }, { partitionKey: "user1" }).fetchAll(); // Explicitly specify partition key in options

    console(`\nActivities for User 1 (purchases only - efficient query):`);
    user1Purchases.forEach(item => console(`  - [${item.userId}] Type: ${item.type}, ID: ${item.id}`));
    console(`Request Charge: ${charge1} RUs`);

    // 3. Inefficient Query (cross-partition query - should be avoided if possible)
    // This query would scan all logical partitions, leading to higher RUs.
    const query2 = `SELECT * FROM c WHERE c.type = "login"`;
    console(`\nActivities of type "login" (INEFFICIENT cross-partition query):`);
    try {
        const { resources: loginActivities, requestCharge: charge2 } = await container.items.query(query2, {
            enableCrossPartitionQuery: true // Required for queries spanning multiple partitions
        }).fetchAll();
        loginActivities.forEach(item => console(`  - [${item.userId}] Type: ${item.type}, ID: ${item.id}`));
        console(`Request Charge: ${charge2} RUs`);
    } catch (error) {
        console(`  Error running cross-partition query without 'enableCrossPartitionQuery': ${error.message}`);
        // In some SDK versions or configurations, cross-partition queries without
        // explicit enablement might throw an error instead of automatically fetching.
    }

    // 4. Point Read (most efficient operation)
    const { resource: specificActivity, requestCharge: charge3 } = await container.item("activity1-user1", "user1").read();
    console(`\nSpecific activity (point read):`);
    console(`  - [${specificActivity.userId}] Type: ${specificActivity.type}, ID: ${specificActivity.id}`);
    console(`Request Charge: ${charge3} RUs`);
}

runPartitionedQueryExample().catch(console.error);
```

#### 5. Go - Integrating with MongoDB API

This Go example demonstrates interacting with Azure Cosmos DB using the MongoDB API. This allows developers to leverage existing MongoDB skills and tools while benefiting from Cosmos DB's global distribution and scalability.

**Scenario:** Storing and querying "Order" documents using the MongoDB Go driver.

**Prerequisites:**

*   Azure Cosmos DB account (MongoDB API endpoint).
*   Go (latest version).
*   MongoDB Go Driver (`go.mongodb.org/mongo-driver/mongo`).

```go
package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"go.mongodb.org/mongo-driver/mongo/readpref"
)

// Replace with your Cosmos DB MongoDB connection string
const connectionString = "YOUR_COSMOS_DB_MONGODB_CONNECTION_STRING"
const databaseName = "OrderDb"
const collectionName = "Orders"
const partitionKey = "customer_id" // For Cosmos DB MongoDB API, partition key needs to be configured during collection creation in Azure Portal

type Order struct {
	ID         string    `bson:"_id,omitempty"`
	CustomerID string    `bson:"customer_id"`
	Amount     float64   `bson:"amount"`
	Item       string    `bson:"item"`
	OrderDate  time.Time `bson:"order_date"`
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Connect to Cosmos DB MongoDB API
	client, err := mongo.Connect(ctx, options.Client().ApplyURI(connectionString))
	if err != nil {
		log.Fatal(err)
	}
	defer func() {
		if err = client.Disconnect(ctx); err != nil {
			log.Fatal(err)
		}
	}()

	// Ping the primary to verify connection
	if err := client.Ping(ctx, readpref.Primary()); err != nil {
		log.Fatal(err)
	}
	fmt.Println("Successfully connected to Azure Cosmos DB (MongoDB API).")

	collection := client.Database(databaseName).Collection(collectionName)

	// 1. Insert an order
	order1 := Order{
		CustomerID: "cust123",
		Amount:     150.75,
		Item:       "Wireless Headphones",
		OrderDate:  time.Now().Add(-24 * time.Hour),
	}
	insertResult, err := collection.InsertOne(ctx, order1)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\nInserted order with ID: %v\n", insertResult.InsertedID)

	order2 := Order{
		CustomerID: "cust456",
		Amount:     25.00,
		Item:       "USB Cable",
		OrderDate:  time.Now().Add(-48 * time.Hour),
	}
	_, err = collection.InsertOne(ctx, order2)
	if err != nil {
		log.Fatal(err)
	}

	order3 := Order{
		CustomerID: "cust123",
		Amount:     300.50,
		Item:       "Gaming Keyboard",
		OrderDate:  time.Now(),
	}
	_, err = collection.InsertOne(ctx, order3)
	if err != nil {
		log.Fatal(err)
	}

	// 2. Find orders for a specific customer (efficient query using partition key)
	fmt.Println("\nOrders for customer_id 'cust123':")
	filter := bson.M{partitionKey: "cust123"}
	cursor, err := collection.Find(ctx, filter)
	if err != nil {
		log.Fatal(err)
	}
	defer cursor.Close(ctx)

	var orders []Order
	if err = cursor.All(ctx, &orders); err != nil {
		log.Fatal(err)
	}
	for _, order := range orders {
		fmt.Printf("  ID: %v, Item: %s, Amount: %.2f, Date: %s\n", order.ID, order.Item, order.OrderDate.Format("2006-01-02"))
	}

	// 3. Update an order
	updateFilter := bson.M{"_id": insertResult.InsertedID}
	update := bson.M{"$set": bson.M{"amount": 160.00}}
	updateResult, err := collection.UpdateOne(ctx, updateFilter, update)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\nUpdated %v document(s)\n", updateResult.ModifiedCount)

	// 4. Delete an order
	deleteFilter := bson.M{"customer_id": "cust456"}
	deleteResult, err := collection.DeleteOne(ctx, deleteFilter)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\nDeleted %v document(s)\n", deleteResult.DeletedCount)

	// Verify deletion
	count, err := collection.CountDocuments(ctx, bson.M{partitionKey: "cust456"})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Remaining orders for 'cust456': %d\n", count)
}
```

### Open-Source Projects for Azure Cosmos DB

Open-source projects significantly enhance the developer experience, facilitate integrations, and extend Cosmos DB's capabilities.

1.  **Azure Cosmos DB .NET SDK**
    *   **Description:** The official .NET client library for Azure Cosmos DB for NoSQL, enabling developers to connect, manage, and perform data operations (CRUD, queries, change feed) with full access to Cosmos DB's advanced features, including vector search.
    *   **GitHub Repository:** [https://github.com/Azure/azure-cosmos-dotnet-v3](https://github.com/Azure/azure-cosmos-dotnet-v3)

2.  **Azure Cosmos DB Desktop Data Migration Tool**
    *   **Description:** A modern, cross-platform command-line tool designed for seamless data migration to and from Azure Cosmos DB. It supports multiple source and sink systems, making it invaluable for developers needing to transfer data efficiently.
    *   **GitHub Repository:** [https://github.com/AzureCosmosDB/data-migration-desktop-tool](https://github.com/AzureCosmosDB/data-migration-desktop-tool)

3.  **Apache Spark Connector for Azure Cosmos DB**
    *   **Description:** This connector facilitates high-performance integration between Apache Spark and Azure Cosmos DB, enabling Spark to read and write data from Cosmos DB. It's crucial for scenarios requiring big data processing, analytics, and machine learning on your globally distributed operational data.
    *   **GitHub Repository:** [https://github.com/Azure/azure-cosmosdb-spark](https://github.com/Azure/azure-cosmosdb-spark)

4.  **Kafka Connectors for Azure Cosmos DB (SQL API)**
    *   **Description:** Offers robust Kafka Connectors for the Azure Cosmos DB SQL API, enabling efficient, real-time data ingress and egress between Kafka topics and Cosmos DB containers. This is vital for building reactive, event-driven applications and microservices that leverage both technologies.
    *   **GitHub Repository:** [https://github.com/Azure/azure-cosmosdb-kafka-connect](https://github.com/Azure/azure-cosmosdb-kafka-connect)

5.  **Azure Cosmos DB Python SDK**
    *   **Description:** The essential Python client library for Azure Cosmos DB for NoSQL, offering comprehensive capabilities to manage and interact with Cosmos DB resources, including CRUD operations, efficient querying, and support for partition keys. It's a core tool for Python developers.
    *   **GitHub Repository:** [https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/cosmos/azure-cosmos](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/cosmos/azure-cosmos)

## Technology Adoption

Azure Cosmos DB is being widely adopted by leading companies for a variety of mission-critical applications requiring low-latency, high availability, and elastic scalability. The latest information from 2025 highlights its growing role in AI-powered applications, alongside its traditional strengths in high-scale operational data.

Here is a list of companies leveraging Azure Cosmos DB and their specific use cases:

*   **Toyota Motor Corporation:** Utilizes Azure Cosmos DB to power its multi-agent AI systems, aiming to enhance productivity within the organization. This adoption is a key component in their modern application architecture.
*   **Jack Henry:** This financial services firm relies on Azure Cosmos DB as a highly scalable, reliable, and fast database solution for its operations, ensuring robust performance for critical financial workloads.
*   **Sayvant:** For clinical documentation solutions, Sayvant uses Azure Cosmos DB to generate faster and more accurate medical charts. This is particularly crucial in high-stakes, fast-paced environments like emergency rooms and urgent care facilities.
*   **Microsoft Copilot:** As an integral part of the Microsoft Copilot development team, Azure Cosmos DB is leveraged in the backend to improve application performance and maintain user engagement for millions of users, underscoring its suitability for large-scale consumer-facing AI applications.
*   **H&R Block:** The tax preparation giant employs Azure Cosmos DB to support both AI-enabled and traditional applications, specifically its mission-critical tier-one tax systems. They benefit from its schemaless data model, distributed replication, and scaling features, including for their AI tax assist features, which have enhanced client satisfaction and conversion rates.
*   **CarMax:** This automotive retailer uses Azure Cosmos DB to track appraisal data within their applications, storing data as vehicles move through valuation processes, demonstrating its ease of interaction and querying for business-critical data.
*   **Walmart:** Azure Cosmos DB underpins Walmart's reordering systems for groceries, serving all transactions with its high performance and scalability, highlighting its role in large-scale e-commerce operations.
*   **JetBlue:** The airline utilizes Azure Cosmos DB to power its fuel management systems, ensuring efficient and reliable operations for a critical aspect of air travel.
*   **OpenAI (ChatGPT):** Serves as the main database for more than 50 various workloads at OpenAI, including ChatGPT itself. Its seamless and elastic scalability, schemaless data model, and built-in vector and hybrid search capabilities make it ideal for rapidly evolving AI applications like large language models, allowing for rapid application development and refactoring.

## References

Here are the top 10 most recent and relevant resources for Azure Cosmos DB, emphasizing the latest features, AI integration, and best practices:

### Official Documentation

1.  **Vector Search in Azure Cosmos DB for NoSQL - Microsoft Learn**
    *   **Description:** This is the authoritative guide to Azure Cosmos DB's native vector indexing and search capabilities, crucial for building modern AI applications like Retrieval Augmented Generation (RAG). It covers how to store, index, and query vector embeddings efficiently at scale.
    *   **Link:** [https://learn.microsoft.com/azure/cosmos-db/nosql/vector-search](https://learn.microsoft.com/azure/cosmos-db/nosql/vector-search)
    *   **Date:** Updated 2025-04-22

2.  **Architecture Best Practices for Azure Cosmos DB for NoSQL - Microsoft Learn**
    *   **Description:** An essential resource for designing robust, scalable, and cost-effective Azure Cosmos DB solutions. It covers critical architectural considerations, including partitioning, indexing, and throughput strategies, aligned with the Azure Well-Architected Framework.
    *   **Link:** [https://learn.microsoft.com/azure/cosmos-db/nosql/architecture-best-practices](https://learn.microsoft.com/azure/cosmos-db/nosql/architecture-best-practices)
    *   **Date:** Updated 2025-08-17

3.  **Query Performance Tips for Azure Cosmos DB SDKs - Microsoft Learn**
    *   **Description:** This documentation provides practical strategies and SDK-specific settings to optimize query performance, reduce Request Unit (RU) consumption, and minimize latency, including recent optimizations like Optimistic Direct Execution (ODE).
    *   **Link:** [https://learn.microsoft.com/azure/cosmos-db/nosql/query-performance-tips-sdk](https://learn.microsoft.com/azure/cosmos-db/nosql/query-performance-tips-sdk)
    *   **Date:** Updated 2025-07-22

4.  **Retrieval Augmented Generation (RAG) in Azure Cosmos DB - Microsoft Learn**
    *   **Description:** A detailed explanation of how to leverage Azure Cosmos DB for RAG scenarios, combining large language models (LLMs) with real-time data retrieval for more accurate and contextually relevant AI responses.
    *   **Link:** [https://learn.microsoft.com/azure/cosmos-db/nosql/retrieval-augmented-generation](https://learn.microsoft.com/azure/cosmos-db/nosql/retrieval-augmented-generation)
    *   **Date:** Updated 2024-12-03

### YouTube Videos

5.  **Everything New in Azure Cosmos DB from Microsoft Build 2025 - ep. 105 - YouTube**
    *   **Description:** This video provides a comprehensive overview of the latest features and advancements announced at Microsoft Build 2025, covering AI integration, smarter search, enhanced developer experiences, and performance upgrades.
    *   **Link:** [https://www.youtube.com/watch?v=D-j0c4G9i_U](https://www.youtube.com/watch?v=D-j0c4G9i_U)
    *   **Date:** 2025-05-28

6.  **Empowering AI Applications with Vector Search in Azure Cosmos DB - YouTube**
    *   **Description:** A practical session explaining RAG, text embeddings, and vector search fundamentals. It includes live demos on how to integrate Azure Cosmos DB with Azure OpenAI for powerful AI experiences.
    *   **Link:** [https://www.youtube.com/watch?v=Yf1e5-8n5hU](https://www.youtube.com/watch?v=Yf1e5-8n5hU)
    *   **Date:** 2024-11-12

7.  **What's New for Java Developers in Azure Cosmos DB (June 2024–2025) - YouTube**
    *   **Description:** This video highlights major updates for Java developers, including seamless AI integration with Spring AI and Langchain4J, Java SDK upgrades, native full-text and hybrid search support, and new indexing options.
    *   **Link:** [https://www.youtube.com/watch?v=MhS0B41l4yY](https://www.youtube.com/watch?v=MhS0B41l4yY)
    *   **Date:** 2025-07-02

### Well-known Technology Blogs & Community Resources

8.  **New Vector Search, Full Text Search, and Hybrid Search Features in Azure Cosmos DB for NoSQL - Microsoft Developer Blogs**
    *   **Description:** The official announcement blog post detailing the General Availability of Vector Search and the Microsoft DiskANN vector index, along with the Public Preview of Full Text and Hybrid Search, significantly enhancing AI retrieval capabilities.
    *   **Link:** [https://devblogs.microsoft.com/cosmosdb/new-vector-search-full-text-search-and-hybrid-search-features-in-azure-cosmos-db-for-nosql/](https://devblogs.microsoft.com/cosmosdb/new-vector-search-full-text-search-and-hybrid-search-features-in-azure-cosmos-db-for-nosql/)
    *   **Date:** 2024-11-19

9.  **Build a RAG application with LangChain and Local LLMs powered by Ollama - Azure Cosmos DB Blog**
    *   **Description:** A hands-on blog post guiding you through building a RAG application using local LLMs with Azure Cosmos DB as a vector database, featuring LangChain integration for embedding, data loading, and vector search.
    *   **Link:** [https://devblogs.microsoft.com/cosmosdb/build-rag-application-langchain-local-llms-ollama-azure-cosmos-db/](https://devblogs.microsoft.com/cosmosdb/build-rag-application-langchain-local-llms-ollama-azure-cosmos-db/)
    *   **Date:** 2025-08-06

10. **AzureCosmosDB/build-2025-search-tips - GitHub**
    *   **Description:** This GitHub repository, directly from Microsoft Build 2025, provides practical tips and best practices, complete with code examples, for optimizing and tuning vector, full-text, and hybrid search in Azure Cosmos DB.
    *   **Link:** [https://github.com/AzureCosmosDB/build-2025-search-tips](https://github.com/AzureCosmosDB/build-2025-search-tips)
    *   **Date:** 2025-05-22

## People Worth Following

Here are the most prominent, relevant, and key contributing people in the Azure Cosmos DB technology domain, worth following on LinkedIn:

1.  **Dharma Shukla**
    *   **Role:** Founder of Azure Cosmos DB and a Distinguished Engineer at Microsoft. Dharma Shukla laid the foundational vision for Cosmos DB as a globally distributed, multi-model database service. His insights into distributed systems design remain highly influential.
    *   **LinkedIn:** [https://www.linkedin.com/in/dharma-shukla-3142271](https://www.linkedin.com/in/dharma-shukla-3142271)

2.  **Kirill Gavrylyuk**
    *   **Role:** Vice President and General Manager for Azure Cosmos DB at Microsoft. Kirill leads the entire product group, overseeing the strategic direction, engineering, and product development for Cosmos DB. He frequently shares updates on the latest features, including AI integration and performance enhancements.
    *   **LinkedIn:** [https://www.linkedin.com/in/kirill-gavrylyuk](https://www.linkedin.com/in/kirill-gavrylyuk)

3.  **Mark Brown**
    *   **Role:** Principal Program Manager on the Azure Cosmos DB team at Microsoft. Mark is deeply involved in ensuring Cosmos DB is developer-friendly, contributing significantly to documentation, samples, and community engagement. He's a key voice on topics like data modeling and partitioning.
    *   **LinkedIn:** [https://www.linkedin.com/in/markjbrown](https://www.linkedin.com/in/markjbrown)

4.  **Deborah Chen**
    *   **Role:** Program Manager for Azure Cosmos DB at Microsoft. Deborah focuses on enhancing the developer experience, including the Azure portal, notebooks, and elasticity features like autoscale and partition merge. She's a frequent speaker on best practices for optimizing Cosmos DB workloads.
    *   **LinkedIn:** [https://www.linkedin.com/in/deborah-chen-39a7385a](https://www.linkedin.com/in/deborah-chen-39a7385a)

5.  **James Codella**
    *   **Role:** Principal Product Manager for Azure Cosmos DB at Microsoft. James focuses on NoSQL Query, vector search, and AI products. His work is crucial for the integration of Cosmos DB into AI applications, particularly with DiskANN for vector search.
    *   **LinkedIn:** [https://www.linkedin.com/in/jamescodella](https://www.linkedin.com/in/jamescodella)

6.  **Nitya Narasimhan**
    *   **Role:** Senior AI Advocate at Microsoft. While her role spans broader AI, Nitya is a prominent voice in the community, frequently discussing and demonstrating how Azure Cosmos DB integrates with AI services and RAG scenarios. Her efforts in developer education are highly valued.
    *   **LinkedIn:** [https://www.linkedin.com/in/nityan](https://www.linkedin.com/in/nityan)

7.  **Leonard Lobel (Lenni)**
    *   **Role:** Microsoft MVP (Data Platform) and CTO/Co-founder of Sleek Technologies. Lenni is a highly active community influencer, speaker, and author who consistently shares deep technical insights and practical guidance on Azure Cosmos DB, with a recent emphasis on its vector database capabilities and AI integration.
    *   **LinkedIn:** [https://www.linkedin.com/in/leonardlobel](https://www.linkedin.com/in/leonardlobel)

8.  **Michael Calvin**
    *   **Role:** CTO at Kinectify. As a technology leader from a company that extensively uses Azure Cosmos DB, Michael provides invaluable real-world insights into building scalable, high-performance, and cost-efficient applications. He often speaks about their architecture and lessons learned.
    *   **LinkedIn:** [https://www.linkedin.com/in/michael-calvin](https://www.linkedin.com/in/michael-calvin)

9.  **Estefani Arroyo**
    *   **Role:** Product Manager for Azure Cosmos DB at Microsoft. Estefani is involved in the development and presentation of tools and features that streamline the developer experience, often focusing on desktop tools and optimizing for performance.
    *   **LinkedIn:** [https://www.linkedin.com/in/estefaniarroyo](https://www.linkedin.com/in/estefaniarroyo)

10. **Tara Bhatia**
    *   **Role:** Product Manager for Azure Cosmos DB at Microsoft. Tara's work focuses on elasticity features such as autoscale, merge, and burst capacity, helping users understand how to achieve better cost and performance. She is a regular speaker at major Microsoft events.
    *   **LinkedIn:** [https://www.linkedin.com/in/tarabhatia](https://www.linkedin.com/in/tarabhatia)