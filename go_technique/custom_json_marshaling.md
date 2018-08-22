# Custom JSON Marshalling
## Default JSON Encoding
Go has a default JSON package that pretty much solves most of our problems. Here's a quick reminder
on how it works
```golang
package main

import "encoding/json"

type Dog struct {
    Name string `json:"name"`
    Age  uint   `json:"age"`
}

func main() {
    bytes, _ := json.Marshal(&Dog{"Loki", 6})
}
```

If we convert the bytes into string, we will see the following.
```
{
    "name": "Loki",
    "age": 6
}
```

Similarly, if we have a JSON text string, we convert it into bytes and we can unmarshal the data 
back into a struct.
```
d := &Dog{}
json.Unmarshal(bytes, d)
```

## Custom JSON Encoding
However, there are times we want to change how data are being shown to users. For example, perhaps 
users don't really want to know the dog age in human years, instead they prefer to see it in dog
years. We can write a custom JSON marshaller to deal with this problem.
```golang
func (d *Dog) MarshalJSON() ([]byte, error) {
    return json.Marshal(&struct{
        Name string `json:"name"`
        Age  uint   `json:"age_in_dog_year"`
    }{
        Name: d.Name,
        Age: a.Age * 7,
    })
}
```

Why does it work? Let's take a look at what Go doc says about `encoding/json`.
> Marshal traverses the value v recursively. If an encountered value implements the Marshaler
> interface and is not a nil pointer, Marshal calls its MarshalJSON method to produce JSON. If no 
> MarshalJSON method is present but the value implements encoding.TextMarshaler instead, Marshal 
> calls its MarshalText method and encodes the result as a JSON string. The nil pointer exception is 
> not strictly necessary but mimics a similar, necessary exception in the behavior of UnmarshalJSON.

When we implement `MarshalJSON` on `Dog` struct, `Dog` becomes a `Marshaler`. Similarly, we can do
the same for `Unmarshal`.
```golang
func (d *Dog) UnmarshalJSON(data []byte) error {
    aux := &struct{
        Name string `json:"name"`
        Age  uint   `json:"age_in_dog_year"`
    }{}

    if err := json.Unmarshal(data, aux); err != nil {
        return err
    }

    d.Name = aux.Name 
    d.Age = aux.Age / 7
    
    return nil
}
```

## Better Approach
Obviously it'd be very cumbersome to write the custom marshal methods if a struct has many fields. 
We can use Golang composition, i.e. struct embedding, to reduce the unnecessary lines of code.
```golang
type Dog struct {
	Name string `json:"name"`
	Age  uint
}

func (d *Dog) MarshalJSON() ([]byte, error) {
	type DogAlias Dog
	return json.Marshal(&struct {
		*DogAlias
		Age uint `json:"age_in_dog_year"`
	}{
		DogAlias: (*DogAlias)(d),
		Age:      d.Age * 7,
	})
}

func (d *Dog) UnmarshalJSON(data []byte) error {
	type DogAlias Dog
	aux := &struct {
		*DogAlias
		Age uint `json:"age_in_dog_year"`
	}{
		DogAlias: (*DogAlias)(d),
	}

	if err := json.Unmarshal(data, aux); err != nil {
		return err
	}

	d.Age = aux.Age / 7

	return nil
}
```

Notice the `Alias` structs, they are for preventing `MarshalJSON` being called infinitely. If we 
declare `aux` with the type `Dog`, when we perform `json.Marshal` inside `MarshalJSON`, `MarshalJSON` 
is called again, and the loop goes on infinitely. The `DogAlias` type has all the fields of the 
original `Dog` type but none of the original methods.