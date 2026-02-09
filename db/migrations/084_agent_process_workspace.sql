CREATE TABLE IF NOT EXISTS agent_processes (
    process_id SERIAL PRIMARY KEY,
    owner_member_id INT NOT NULL,
    process_label VARCHAR(160) NOT NULL,
    process_description TEXT,
    process_status VARCHAR(24) NOT NULL DEFAULT 'active',
    completion_percent REAL NOT NULL DEFAULT 0,
    steps_total INT NOT NULL DEFAULT 0,
    steps_completed INT NOT NULL DEFAULT 0,
    process_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP,
    CONSTRAINT process_owner_member_link
        FOREIGN KEY(owner_member_id)
        REFERENCES member_data(member_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_agent_processes_owner_status
    ON agent_processes (owner_member_id, process_status, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_processes_updated_at
    ON agent_processes (updated_at DESC);

CREATE TABLE IF NOT EXISTS agent_process_steps (
    step_id SERIAL PRIMARY KEY,
    process_id INT NOT NULL,
    step_order INT NOT NULL,
    step_label VARCHAR(240) NOT NULL,
    step_details TEXT,
    step_status VARCHAR(24) NOT NULL DEFAULT 'pending',
    is_required BOOLEAN NOT NULL DEFAULT TRUE,
    step_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP,
    CONSTRAINT process_step_process_link
        FOREIGN KEY(process_id)
        REFERENCES agent_processes(process_id)
        ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_process_steps_unique_order
    ON agent_process_steps (process_id, step_order);

CREATE INDEX IF NOT EXISTS idx_agent_process_steps_status
    ON agent_process_steps (process_id, step_status, step_order);
