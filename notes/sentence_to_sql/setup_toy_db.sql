PRAGMA encoding = 'utf-8';
DROP TABLE IF EXISTS People;
CREATE TABLE People(
       id INTEGER PRIMARY KEY,
       "given_nameLabel.value" TEXT NOT NULL,
       "family_nameLabel.value" TEXT NOT NULL,
       "itemLabel.value" TEXT NOT NULL,
       "date_of_birthLabel.value" DATETIME,
       "date_of_deathLabel.value" DATETIME);
INSERT INTO People (
    "given_nameLabel.value",
    "family_nameLabel.value",
    "itemLabel.value",
    "date_of_birthLabel.value",
    "date_of_deathLabel.value"
) VALUES
    ('Paul', 'Dirac', 'Paul Dirac', '1902-08-08', '1984-10-20'),
    ('Lee', 'Yew', 'Lee Kuan Yew', '1923-09-16', '2015-03-23'),
    ('Sun', 'Yat-sen', 'Sun Yat-sen', '1866-11-12', '1925-03-12'),
    ('Mohandas', 'Gandhi', 'Mahatma Gandhi', '1869-10-02', '1948-01-30'),
    ('Neville', 'Chamberlain', 'Neville Chamberlain', '1869-03-18', '1940-11-09'),
    ('Clement', 'Attlee', 'Clement Attlee', '1883-01-03', '1967-10-08'),
    ('Anna', 'Roosevelt', 'Eleanor Roosevelt', '1884-10-11', '1962-11-07'),
    ('Martha', 'Stewart', 'Martha Stewart', '1941-08-03', NULL),
    ('Katalin', 'Kariko', 'Katalin Kariko', '1955-01-17', NULL),
    ('Kelly', 'Clarkson', 'Kelly Clarkson', '1982-04-24', NULL),
    ('Augusta', 'King', 'Ada Lovelace', '1815-12-10', '1852-11-27'),
    ('Charles', 'Babbage', 'Charles Babbage', '1791-12-26', '1871-10-18')
;