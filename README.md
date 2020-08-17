# Semantic search tool to find relevant (similar or related) terms.
Description of each file:
# Data: 
The data used for this project was extracted from a Scopus. It covers the 14 different categories of articles within the domain of Business literature – HRM with the publication year from 2000 to 2019.  The total number of extracted data was 9820 and categories are shown as follows:
1.	Academy of Management Journal
2.	Journal of Management
3.	European Journal of Work and organization psychology
4.	Human Resource Management Journal
5.	Human relations
6.	Human Resource Management
7.	International Journal of Human Resource Management
8.	Journal of applied psychology
9.	Journal of Management Studies
10.	Personnel Psychology
11.	Journal of organizational behavior
12.	Leadership Quarterly
13.	Journal of Occupational and Organizational Psychology
14.	Organizational Behavior and Human Decision Processes

# Pre-processing:
Each category of articles was pre-processed separately but saved as a document one sentence per line to form a larger data set. The pre-processed data is available in the data folder. Pre-processed data was not uploaded on the GitHub because of its size. It is saved locally.

# Lemmatization: 
Pre-processed data were processed for lemmatization and the result was written line by line to the new file.
Training_Model: It consists of a line-based iterator that reads the lemmatized file one line at a time instead of reading everything in memory at once. We set the threshold to remove a certain section of vocabulary. The lemmatized data was trained with the Word2Vec model and the result is saved in the data folder. Finally, the model was loaded in a “semantic-search” repository for the deployment of the application.
