# Personalized Content Recommendation System

Welcome! This project is a fully functional, interest-based content recommendation system. The goal is simple: to intelligently suggest the top 3 most relevant posts for any given user.

This isn't just a random selection. The system dives deep into user profiles, past engagement, and the specific attributes of each post to create a truly personalized discovery experience. The entire recommendation engine is wrapped in a clean, interactive web application built with Streamlit.

[Demo of the Streamlit Application]([https://i.imgur.com/g0j8z9G.gif](https://content-recommendation-system.streamlit.app/))

## How it Works: My Approach

To deliver high-quality recommendations, I built a **hybrid model**. This is the same approach used by major tech companies, as it combines the strengths of multiple strategies to create a result that is both accurate and personalized. Our model has two core components:

### 1. Content-Based Filtering (The "Interest Matcher")

The first part of the engine focuses on a simple but powerful idea: *"If you like posts about 'sports', you should see more posts about 'sports'."*

To make this work, we need to teach the computer how to understand and compare text. I used a standard, industry-proven technique:

* **TF-IDF (Term Frequency-Inverse Document Frequency):** This sounds complex, but it's just a clever way to convert text (like user interests and post tags) into numerical vectors. It identifies which words are truly important by giving more weight to tags that are specific (e.g., 'literature') and less weight to ones that are very common.
* **Cosine Similarity:** Once both users and posts are represented as vectors, we can calculate the "angle" between them. A smaller angle (a score closer to 1) means the user's interests and the post's tags are a strong match!

### 2. User-Based Heuristic (The "Personalization Layer")

A user's interests don't tell the whole story. Some users are naturally more active and engaged than others. The `past_engagement_score` is a fantastic piece of data that helps us understand this. By incorporating this score, we can fine-tune the recommendations, giving a slight boost to content for our most active users.

### 3. The Hybrid Score 

The final step is to combine these two signals into a single, powerful **Recommendation Score**. I used a simple weighted formula:

$$\text{Recommendation Score} = (0.7 \times \text{Interest Match Score}) + (0.3 \times \text{User Engagement Score})$$

This formula prioritizes a strong interest match (70% weight) while still allowing a user's overall engagement level to influence the final ranking (30% weight). Of course, we always make sure to filter out any posts the user has already seen!

---

## A Look Under the Hood: Data Insights

To give you a better sense of the process, here’s a peek at the data and how our model sees it.

#### Raw Data Samples

We start with three simple datasets:

**Users.csv**
| user_id | age | gender | top_3_interests | past_engagement_score |
| :--- | :-: | :---: | :--- | :---: |
| U1 | 24 | F | sports, art, gaming | 0.61 |
| U2 | 32 | F | travel, food, fashion | 0.93 |
| U3 | 28 | Other | sports, travel, fashion | 0.40 |
| U4 | 25 | M | fashion, music, tech | 0.53 |
| U5 | 24 | M | fashion, food, fitness | 0.80 |

**Posts.csv**
| post_id | creator_id | content_type | tags |
| :--- | :---: | :---: | :--- |
| P1 | U44 | video | sports, food |
| P2 | U26 | video | music, travel |
| P3 | U32 | text | sports, travel |
| P4 | U6 | image | music, gaming |
| P5 | U32 | image | food, fashion |

**Engagements.csv**
| user_id | post_id | engagement |
| :--- | :---: | :---: |
| U1 | P52 | 1 |
| U1 | P44 | 0 |
| U1 | P1 | 1 |
| U1 | P4 | 1 |
| U1 | P65 | 0 |

#### The Model's Viewpoint

After the TF-IDF process, our text is transformed into numerical matrices. The shapes tell us we have 50 users and 100 posts, and have identified 10 unique interest/tag features across the dataset.

| Matrix Description | Shape |
| :--- | :--- |
| User-Interests Matrix | (50, 10) |
| Post-Tags Matrix | (100, 10) |

Finally, the cosine similarity calculation gives us a score for every possible user-post pair, resulting in our **Interest Match Score Matrix**.

| Matrix Description | Shape | Sample Value Interpretation |
| :--- | :--- | :--- |
| Interest Match Score Matrix | (50, 100) | The value at `[0, 0]` (0.4199) is the match between User `U1` and Post `P1`. |

---

## How to Run the App

Getting the recommendation system up and running on your local machine is easy.

1.  **Set up the Environment:**
    * Clone the repository.
    * Create and activate a Python virtual environment.

2.  **Install Dependencies:**
    * Run the following command to install all the necessary libraries:
        ```bash
        pip install -r requirements.txt
        ```

3.  **Launch the App:**
    * Open your terminal, navigate to the project's root folder, and run:
        ```bash
        streamlit run app/app.py
        ```
    * A new tab will open in your web browser with the interactive application!

## Evaluation and Metrics

Because this is a recommendation task, traditional metrics like accuracy aren't the best fit. Instead, the most effective evaluation is **qualitative**.

The true test is to select a user from the dropdown in the app and observe the results. For instance, if you select a user who is interested in `sports, art, gaming`, the top 3 recommended posts should have tags that clearly align with these interests. This "common sense" check is often the most important validation for a recommendation system.

---

## Future Directions & Possible Extensions

This project provides a very strong foundation, but there are several exciting ways it could be extended in a real-world setting:

1.  **Incorporate Collaborative Filtering:** The next logical step would be to add a "users who liked this also liked..." feature. By analyzing patterns across *all* users, we can uncover recommendations that aren't immediately obvious from a user's stated interests. Techniques like **Matrix Factorization (SVD)** would be perfect for this.

2.  **Leverage Deeper Content Analysis:** Instead of just tags, we could use Natural Language Processing (NLP) to analyze the *full text* of the posts. This would allow for a much more nuanced understanding of the content.

3.  **Implement A/B Testing:** In a live environment, the ultimate test is to deploy different versions of the recommendation model to different users and measure which one leads to higher engagement (more clicks, likes, etc.). This provides concrete data on what truly works best.

