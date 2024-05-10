#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind<'a> {
    Ident(&'a str),
    Colon,
    Whitespace,
    AssignmentOp,
    Semicolon,
    IntegerLiteral(&'a str),
    CharLiteral(&'a str),
    LogicalAnd,
    LogicalOr,
    PostIncrement,
    PostDecrement,
    LessThanOrEqual,
    GreaterThanOrEqual,
    Equal,
    NotEqual,
    //UnaryNegation, // not sure about this one
    LogicalNot,
    Exponentiation,
    Multiply,
    Divide,
    Modulo,
    Add,
    Subtract,
    LessThan,
    GreaterThan,
    LeftCurlyBrace,
    RightCurlyBrace,
    LeftParen,
    RightParen,
    LeftBracket,
    RightBracket,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpanBound {
    line: usize,
    col: usize,
}

impl SpanBound {
    pub fn new(line: usize, col: usize) -> Self {
        Self { line, col }
    }
}

impl From<(usize, usize)> for SpanBound {
    fn from(value: (usize, usize)) -> Self {
        Self::new(value.0, value.1)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    start: SpanBound,
    end: SpanBound,
}

impl Span {
    pub fn new(start: SpanBound, end: SpanBound) -> Self {
        Self { start, end }
    }
}

impl From<(SpanBound, SpanBound)> for Span {
    fn from(value: (SpanBound, SpanBound)) -> Self {
        Self::new(value.0, value.1)
    }
}

impl From<((usize, usize), (usize, usize))> for Span {
    fn from(value: ((usize, usize), (usize, usize))) -> Self {
        Self::new(value.0.into(), value.1.into())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token<'a> {
    kind: TokenKind<'a>,
    span: Span,
}

struct Cursor<'a> {
    current_line: usize,
    current_col: usize,
    remaining_src: &'a str,
}

impl<'a> Cursor<'a> {
    pub fn new(src: &'a str, line_number: usize) -> Self {
        Cursor {
            current_line: line_number,
            current_col: 0,
            remaining_src: src,
        }
    }

    pub fn get_next_token(&mut self) -> Option<Token<'a>> {
        if let Some(token) = self.get_ident() {
            return Some(token);
        }
        if let Some(token) = self.get_literal() {
            return Some(token);
        }
        if let Some(token) = self.get_whitespace() {
            return Some(token);
        }
        if let Some(token) = self.get_multichar_op() {
            return Some(token);
        }
        self.get_simple_token()
    }

    fn consume_src(&mut self, len: usize) -> Span {
        let start = SpanBound::new(self.current_line, self.current_col);
        let end = SpanBound::new(self.current_line, self.current_col + len);
        self.current_col += len;
        self.remaining_src = &self.remaining_src[len..];
        Span::new(start, end)
    }

    fn get_multichar_op(&mut self) -> Option<Token<'a>> {
        let mut src_iter = self.remaining_src.chars().into_iter();
        let kind = match (src_iter.next()?, src_iter.next()?) {
            ('+', '+') => Some(TokenKind::PostIncrement),
            ('-', '-') => Some(TokenKind::PostDecrement),
            ('<', '=') => Some(TokenKind::LessThanOrEqual),
            ('>', '=') => Some(TokenKind::GreaterThanOrEqual),
            ('=', '=') => Some(TokenKind::Equal),
            ('!', '=') => Some(TokenKind::NotEqual),
            ('&', '&') => Some(TokenKind::LogicalAnd),
            ('|', '|') => Some(TokenKind::LogicalOr),
            _ => None,
        };
        if let Some(kind) = kind {
            let span = self.consume_src(2);
            Some(Token { kind, span })
        } else {
            None
        }
    }

    fn get_simple_token(&mut self) -> Option<Token<'a>> {
        let token_kind = match self.remaining_src.chars().next()? {
            ':' => Some(TokenKind::Colon),
            '=' => Some(TokenKind::AssignmentOp),
            ';' => Some(TokenKind::Semicolon),
            '!' => Some(TokenKind::LogicalNot),
            '^' => Some(TokenKind::Exponentiation),
            '*' => Some(TokenKind::Multiply),
            '/' => Some(TokenKind::Divide),
            '%' => Some(TokenKind::Modulo),
            '+' => Some(TokenKind::Add),
            '-' => Some(TokenKind::Subtract),
            '<' => Some(TokenKind::LessThan),
            '>' => Some(TokenKind::GreaterThan),
            '{' => Some(TokenKind::LeftCurlyBrace),
            '}' => Some(TokenKind::RightCurlyBrace),
            '(' => Some(TokenKind::LeftParen),
            ')' => Some(TokenKind::RightParen),
            '[' => Some(TokenKind::LeftBracket),
            ']' => Some(TokenKind::RightBracket),
            _ => None,
        }?;
        let span = self.consume_src(1);
        Some(Token {
            kind: token_kind,
            span,
        })
    }

    fn get_whitespace(&mut self) -> Option<Token<'a>> {
        let mut src_iter = self.remaining_src.chars().into_iter();
        let mut whitespace_len = 0;
        while let Some(ch) = src_iter.next() {
            if ch.is_ascii_whitespace() {
                whitespace_len += 1;
            } else {
                break;
            }
        }

        if whitespace_len > 0 {
            let span = self.consume_src(whitespace_len);
            Some(Token {
                kind: TokenKind::Whitespace,
                span,
            })
        } else {
            None
        }
    }

    fn get_ident(&mut self) -> Option<Token<'a>> {
        if Self::is_valid_ident_start(self.remaining_src.chars().into_iter().next()?) {
            let mut ident_len = 1;
            for ch in self.remaining_src[1..].chars().into_iter() {
                if Self::is_valid_ident(ch) {
                    ident_len += 1;
                } else {
                    break;
                }
            }
            let ident_src = &self.remaining_src[..ident_len];
            let ident_span = self.consume_src(ident_len);
            Some(Token {
                kind: TokenKind::Ident(ident_src),
                span: ident_span,
            })
        } else {
            None
        }
    }

    fn is_valid_ident_start(ch: char) -> bool {
        ch.is_ascii_alphabetic() || ch == '_'
    }

    fn is_valid_ident(ch: char) -> bool {
        ch.is_ascii_alphanumeric() || ch == '_'
    }

    fn get_literal(&mut self) -> Option<Token<'a>> {
        if let Some(token) = self.get_int_literal() {
            return Some(token);
        } else {
            if let Some(token) = self.get_char_literal() {
                return Some(token);
            }
        }
        None
    }

    fn get_int_literal(&mut self) -> Option<Token<'a>> {
        // not handling radix yet
        let mut literal_len = 0;
        for ch in self.remaining_src.chars().into_iter() {
            if ch.is_ascii_digit() {
                literal_len += 1;
            } else {
                break;
            }
        }

        if literal_len > 0 {
            let literal_src = &self.remaining_src[..literal_len];
            let literal_span = self.consume_src(literal_len);
            Some(Token {
                kind: TokenKind::IntegerLiteral(literal_src),
                span: literal_span,
            })
        } else {
            None
        }
    }

    fn get_char_literal(&mut self) -> Option<Token<'a>> {
        let mut src_iter = self.remaining_src.chars().into_iter();
        if src_iter.next()? == '\'' {
            if src_iter.skip(1).next()? == '\'' {
                let literal_src = &self.remaining_src[..3];
                let literal_span = self.consume_src(3);
                Some(Token {
                    kind: TokenKind::CharLiteral(literal_src),
                    span: literal_span,
                })
            } else {
                None
            }
        } else {
            None
        }
    }
}

pub fn scan(source: &str) -> Vec<Token> {
    let mut tokens = vec![];
    let mut line_number = 0;
    let mut total_source_consumed = 0;

    loop {
        let remaining_src = &source[total_source_consumed..];
        if let Some(newline) = remaining_src.find('\n') {
            let (newline_idx, newline_len) = if let Some(cr) = remaining_src[..newline].find('\r') {
                (cr, 2)
            } else {
                (newline, 1)
            };
            let mut cursor = Cursor::new(&remaining_src[..newline_idx], line_number);
            while let Some(token) = cursor.get_next_token() {
                tokens.push(token);
            }
            tokens.push(Token {
                kind: TokenKind::Whitespace,
                span: Span::new(
                    SpanBound::new(cursor.current_line, cursor.current_col),
                    SpanBound::new(cursor.current_line + 1, 0),
                ),
            });

            total_source_consumed += newline_idx + newline_len;
        } else if !remaining_src.is_empty() {
            let mut cursor = Cursor::new(&remaining_src, line_number);
            while let Some(token) = cursor.get_next_token() {
                tokens.push(token);
            }
            total_source_consumed += remaining_src.len();
        } else {
            break;
        }
        line_number += 1;
    }

    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integer_initialization() {
        let src = "x: integer = 65;";
        let tokens = scan(src);
        let expected_tokens: Vec<Token> = vec![
            Token {
                kind: TokenKind::Ident("x"),
                span: Span::from(((0, 0), (0, 1))),
            },
            Token {
                kind: TokenKind::Colon,
                span: Span::from(((0, 1), (0, 2))),
            },
            Token {
                kind: TokenKind::Whitespace,
                span: Span::from(((0, 2), (0, 3))),
            },
            Token {
                kind: TokenKind::Ident("integer"),
                span: Span::from(((0, 3), (0, 10))),
            },
            Token {
                kind: TokenKind::Whitespace,
                span: Span::from(((0, 10), (0, 11))),
            },
            Token {
                kind: TokenKind::AssignmentOp,
                span: Span::from(((0, 11), (0, 12))),
            },
            Token {
                kind: TokenKind::Whitespace,
                span: Span::from(((0, 12), (0, 13))),
            },
            Token {
                kind: TokenKind::IntegerLiteral("65"),
                span: Span::from(((0, 13), (0, 15))),
            },
            Token {
                kind: TokenKind::Semicolon,
                span: Span::from(((0, 15), (0, 16))),
            },
        ];

        assert_eq!(expected_tokens, tokens);
    }

    #[test]
    fn test_char_initialization() {
        let src = "y: char = 'A';";
        let tokens = scan(src);
        let expected_tokens: Vec<Token> = vec![
            Token {
                kind: TokenKind::Ident("y"),
                span: Span::from(((0, 0), (0, 1))),
            },
            Token {
                kind: TokenKind::Colon,
                span: Span::from(((0, 1), (0, 2))),
            },
            Token {
                kind: TokenKind::Whitespace,
                span: Span::from(((0, 2), (0, 3))),
            },
            Token {
                kind: TokenKind::Ident("char"),
                span: Span::from(((0, 3), (0, 7))),
            },
            Token {
                kind: TokenKind::Whitespace,
                span: Span::from(((0, 7), (0, 8))),
            },
            Token {
                kind: TokenKind::AssignmentOp,
                span: Span::from(((0, 8), (0, 9))),
            },
            Token {
                kind: TokenKind::Whitespace,
                span: Span::from(((0, 9), (0, 10))),
            },
            Token {
                kind: TokenKind::CharLiteral("'A'"),
                span: Span::from(((0, 10), (0, 13))),
            },
            Token {
                kind: TokenKind::Semicolon,
                span: Span::from(((0, 13), (0, 14))),
            },
        ];

        assert_eq!(expected_tokens, tokens);
    }
}
