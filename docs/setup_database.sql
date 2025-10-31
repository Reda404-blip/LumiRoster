CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_name TEXT NOT NULL,
    status TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Insert some sample data
INSERT INTO attendance (student_name, status) VALUES
    ('John Doe', 'Present'),
    ('Jane Smith', 'Absent'),
    ('Mike Johnson', 'Present');