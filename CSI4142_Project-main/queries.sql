
SELECT
    CASE
        WHEN ef.count_words <= 12 THEN '0-12'
        WHEN ef.count_words <= 16 THEN '13-16'
        ELSE '16+'
    END AS word_count_range,
    COUNT(DISTINCT ef.email_id) AS email_count,
    COUNT(DISTINCT CASE WHEN ef.email_id = ef.email_id AND ef1.spam_classification = true THEN ef.email_id END) AS spam_count
FROM
    emailfeatures ef
JOIN
    email_flag ef1
ON
    ef.email_id = ef1.email_id
GROUP BY
    word_count_range
ORDER BY
    word_count_range;

------------------------------------------------------------------------------------------------------------------------------------------------
SELECT
    COUNT(DISTINCT ef.email_id) AS email_count,
    COUNT(DISTINCT CASE WHEN ef.email_id = ef.email_id AND ef1.spam_classification = true THEN ef.email_id END) AS spam_count
FROM
    emailfeatures ef
JOIN
    email_flag ef1
ON
    ef.email_id = ef1.email_id;
------------------------------------------------------------------------------------------------------------------------------------------------
SELECT
    ef1.email_id,
    ef1.messages,
    ef1.count_words,
    ef1.count_characters,
    ef1.count_characters_no_space,
    ef1.avg_word_length,
    ef1.count_digits,
    ef1.count_numbers,
    ef1.noun_count,
    ef1.aux_count,
    ef1.verb_count,
    ef1.adj_count,
    ef1.ner,
    ef1.no_of_spelling_mistakes
FROM
    email_flag ef
JOIN
    emailfeatures ef1
ON
    ef.email_id = ef1.email_id
WHERE
    ef.spam_classification = 'true';

------------------------------------------------------------------------------------------------------------------------------------------------
SELECT
    ef1.email_id,
    ef1.messages,
    ef1.count_words,
    ef1.count_characters,
    ef1.count_characters_no_space,
    ef1.avg_word_length,
    ef1.count_digits,
    ef1.count_numbers,
    ef1.noun_count,
    ef1.aux_count,
    ef1.verb_count,
    ef1.adj_count,
    ef1.ner,
    ef1.no_of_spelling_mistakes
FROM
    email_flag ef
JOIN
    emailfeatures ef1
ON
    ef.email_id = ef1.email_id
WHERE
    ef.spam_classification = 'true'
    AND ef1.count_words > 10;


------------------------------------------------------------------------------------------------------------------------------------------------
SELECT
    ef1.email_id,
    ef1.messages,
    ef1.count_words,
    ef1.count_characters,
    ef1.count_characters_no_space,
    ef1.avg_word_length,
    ef1.count_digits,
    ef1.count_numbers,
    ef1.noun_count,
    ef1.aux_count,
    ef1.verb_count,
    ef1.adj_count,
    ef1.ner,
    ef1.no_of_spelling_mistakes
FROM
    email_flag ef
JOIN
    emailfeatures ef1
ON
    ef.email_id = ef1.email_id
WHERE
    ef.spam_classification = 'false'
    AND ef1.count_characters < 30;


------------------------------------------------------------------------------------------------------------------------------------------------
SELECT
    email_id,
    messages,
    count_characters
FROM
    emailfeatures
WHERE
count_characters > 100;


------------------------------------------------------------------------------------------------------------------------------------------------
SELECT
    noun_count,
    AVG(count_characters) AS avg_characters
FROM
    emailfeatures
WHERE
    count_characters > 50
GROUP BY

------------------------------------------------------------------------------------------------------------------------------------------------
SELECT
    ner,
    AVG(avg_word_length) AS avg_word_length
FROM
    emailfeatures
WHERE
    noun_count > 1
GROUP BY
    Ner;

------------------------------------------------------------------------------------------------------------------------------------------------
SELECT
    adj_count,
    verb_count,
    COUNT(email_id) AS total_emails,
    AVG(count_characters) AS avg_characters
FROM
    emailfeatures
WHERE
    count_characters > 50
GROUP BY
    adj_count,
    verb_count
ORDER BY
    adj_count,
    verb_count;

------------------------------------------------------------------------------------------------------------------------------------------------
SELECT
    email_id,
    messages,
    count_characters
FROM
    emailfeatures
ORDER BY
    count_characters DESC
LIMIT
    100;


------------------------------------------------------------------------------------------------------------------------------------------------

SELECT
    email_id,
    messages,
    count_characters,
    RANK() OVER (ORDER BY count_characters DESC) AS ranking
FROM
    emailfeatures;

------------------------------------------------------------------------------------------------------------------------------------------------
SELECT
    email_id,
    messages,
    avg_word_length,
    AVG(avg_word_length) OVER () AS avg_word_length_all_emails
FROM
    Emailfeatures;


------------------------------------------------------------------------------------------------------------------------------------------------

