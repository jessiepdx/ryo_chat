CREATE TABLE IF NOT EXISTS proposals (
    proposal_id SERIAL PRIMARY KEY,
    submitted_from TEXT,
    project_title TEXT,
    project_description TEXT,
    filename TEXT,
    submit_date timestamp
);
