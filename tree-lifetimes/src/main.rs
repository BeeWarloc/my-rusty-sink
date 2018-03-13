extern crate typed_arena;
extern crate smallvec;

use std::collections::HashMap;
use std::fmt;
use std::cell::RefCell;

#[derive(Copy, Clone, Debug)]
enum BinOp {
    Add, Sub, Mul, Div
}

#[derive(Copy, Clone, Debug)]
enum UnOp {
    Negate
}

type ExprRef<'a> = &'a Expr<'a>;

struct Expr<'a> {
    data: ExprData<'a>,
    ty: Option<TypeRef<'a>>
}

impl<'a> std::fmt::Debug for Expr<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.data)?;
        if let Some(ty) = self.ty {
            write!(f, ":{:?}", ty)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
enum ExprData<'a> {
    Integer(u64),
    Binary(BinOp, ExprRef<'a>, ExprRef<'a>),
    Unary(UnOp, ExprRef<'a>)
}

type TypeRef<'t> = &'t Type<'t>;
// smallvec is not working
//type TypeVec<'t> = smallvec::SmallVec<[TypeRef<'t>; 4]>;
type TypeVec<'t> = Vec<TypeRef<'t>>;

// #[derive(Debug)]
// enum TypeVec<'t> {
//     Zero,
//     One(TypeRef<'t>),
//     Two(TypeRef<'t>),
//     More(Vec<TypeRef<'t>>)
// }

#[derive(Debug)]
enum Type<'t> {
    Named(String),
    GenericInstance(TypeRef<'t>, TypeVec<'t>)
}


struct TypeContext<'t> {
    types: typed_arena::Arena<Type<'t>>,
    map: RefCell<HashMap<&'t str, TypeRef<'t>>>
}

impl<'t> TypeContext<'t> {
    pub fn new() -> TypeContext<'t> {
        TypeContext { types: typed_arena::Arena::new(), map: RefCell::new(HashMap::new()) }
    }

    pub fn named(&'t self, name: &str) -> TypeRef<'t> {
        if let Some(entry) = self.map.borrow().get(name) {
            return entry;
        }

        let name = name.to_string();
        let ty: &Type = self.types.alloc(Type::Named(name));
        if let &Type::Named(ref s) = ty {
            self.map.borrow_mut().insert(s.as_str(), ty);
        }
        ty
    }
}

struct NodeFactory<'a, 't> {
    nodes: typed_arena::Arena<Expr<'a>>,
    tc: &'t TypeContext<'t>
}

impl<'a, 't> NodeFactory<'a, 't> where 't: 'a {
    pub fn new(tc: &'t TypeContext<'t>) -> NodeFactory<'a, 't> {
        NodeFactory {
            nodes: typed_arena::Arena::new(),
            tc
        }
    }
}

impl<'a, 't> NodeFactory<'a, 't> {
    pub fn unary(&'a self, op: UnOp, node: ExprRef<'a>) -> ExprRef {
        self.node(ExprData::Unary(op, node), None)
    }

    pub fn integer(&'a self, value: u64) -> ExprRef {
        self.node(ExprData::Integer(value), None)
    }

    pub fn bin(&'a self, op: BinOp, left: ExprRef<'a>, right: ExprRef<'a>) -> ExprRef {
        self.node(ExprData::Binary(op, left, right), None)
    }

    fn node(&'a self, data: ExprData<'a>, ty: Option<TypeRef<'a>>) -> ExprRef<'a> {
        self.nodes.alloc(Expr { data, ty })
    }
}

fn main() {
    let tc = TypeContext::new();
    {
        let f = NodeFactory::new(&tc);
        let l = f.integer(1);
        let r = f.integer(2);
        let added = f.bin(BinOp::Add, f.integer(1), f.unary(UnOp::Negate, f.integer(222)));
        println!("{:?}", added);
    }
}
