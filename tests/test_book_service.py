from unittest.mock import patch, mock_open
import pytest
from app.services.book_service import BookService
import pandas as pd

# Test Initialization
def test_initialization():
    service = BookService()
    assert service.data_dir is not None
    assert service.books_file.endswith("books.json")
    assert service.df_file.endswith("books_df.pkl")

# Test collect_books Method
@patch("app.services.book_service.requests.get")
def test_collect_books_default(mock_get):
    mock_get.return_value.json.return_value = {"items": []}
    service = BookService()
    result = service.collect_books()
    assert result == {"items": []}

@patch("app.services.book_service.requests.get")
def test_collect_books_custom_query(mock_get):
    mock_get.return_value.json.return_value = {"items": ["book1", "book2"]}
    service = BookService()
    result = service.collect_books(query="python", max_results=2)
    assert result == {"items": ["book1", "book2"]}

@patch("app.services.book_service.requests.get")
def test_collect_books_pagination(mock_get):
    mock_get.return_value.json.side_effect = [{"items": ["book1"]}, {"items": ["book2"]}]
    service = BookService()
    result = service.collect_books(max_results=41)
    assert result == {"items": ["book1", "book2"]}

@patch("app.services.book_service.requests.get")
def test_collect_books_api_error(mock_get):
    mock_get.side_effect = Exception("API Error")
    service = BookService()
    with pytest.raises(Exception):
        service.collect_books()

# Test clean_data Method
def test_clean_data():
    service = BookService()
    books = {"items": [{"id": "1", "volumeInfo": {"title": "Test Book", "description": "A test book description."}}]}
    result = service.clean_data(books)
    assert len(result) == 1
    assert result[0]["id"] == "1"
    assert "test book description" in result[0]["processed_description"]

# Test _clean_text Method
def test_clean_text():
    service = BookService()
    cleaned_text = service._clean_text("This is a test, with punctuation!")
    assert cleaned_text == "test punctuation"

# Test load_dataframe Method
@patch("os.path.exists")
@patch("pandas.read_pickle")
def test_load_dataframe_exists(mock_read_pickle, mock_exists):
    mock_exists.return_value = True
    mock_read_pickle.return_value = pd.DataFrame()
    service = BookService()
    df = service.load_dataframe()
    assert isinstance(df, pd.DataFrame)