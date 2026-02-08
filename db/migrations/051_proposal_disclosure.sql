CREATE TABLE IF NOT EXISTS proposal_disclosure (
    disclosure_id SERIAL PRIMARY KEY,
    user_id INT,
    proposal_id INT,
    agreement_date timestamp
);
