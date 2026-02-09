CREATE TABLE IF NOT EXISTS member_outbox (
    outbox_id SERIAL PRIMARY KEY,
    sender_member_id INT,
    target_member_id INT NOT NULL,
    target_username VARCHAR(96),
    delivery_channel VARCHAR(24) NOT NULL DEFAULT 'telegram',
    message_text TEXT NOT NULL,
    delivery_status VARCHAR(24) NOT NULL DEFAULT 'queued',
    process_id INT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    delivered_at TIMESTAMP,
    failure_reason TEXT,
    CONSTRAINT outbox_sender_member_link
        FOREIGN KEY(sender_member_id)
        REFERENCES member_data(member_id)
        ON DELETE SET NULL,
    CONSTRAINT outbox_target_member_link
        FOREIGN KEY(target_member_id)
        REFERENCES member_data(member_id)
        ON DELETE CASCADE,
    CONSTRAINT outbox_process_link
        FOREIGN KEY(process_id)
        REFERENCES agent_processes(process_id)
        ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_member_outbox_sender_status
    ON member_outbox (sender_member_id, delivery_status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_member_outbox_target_status
    ON member_outbox (target_member_id, delivery_status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_member_outbox_process
    ON member_outbox (process_id, created_at DESC);
