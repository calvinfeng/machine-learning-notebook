# Assign Pointer
Let's say we have a struct that has pointer fields.
```golang
type Dog struct {
    Name string
    Nickname *string
    Age uint
}
```

When we create an *instance* of `Dog`, we typically do the following.
```golang
nn := "Give No Fucks"
d := &Dog{
    Name: "Loki",
    Nickname: &nn,
    Age: 7,
}
```

However, that can be a little annoying if there are many pointer fields. We can solve this problem 
by abstracting the variable assignment and pointe address extraction away. The abstraction will 
involve using `reflect`. 
```golang
func Ptr(val interface{}) interface{} {
    // Create a pointer to zero value of same type as val
    p := reflect.New(reflect.TypeOf(val))
    
    // Copy the underlying value
    p.Elem().Set(reflect.ValueOf(val))
    
    // Return interface assertable to pointer
    return p.Interface()
} 
```

Now we can condense it into one line.
```golang
d := &Dog{
    Name: "Loki",
    Nickname: Ptr("Give No Fucks").(*string),
    Age: 7,
}
```
