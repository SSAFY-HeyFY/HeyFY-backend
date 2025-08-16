package com.ssafy.ssashinsa.heyfy.authentication.ulid;

import com.github.f4b6a3.ulid.Ulid;
import jakarta.persistence.AttributeConverter;
import jakarta.persistence.Converter;

@Converter(autoApply = true)
public class UlidConverter implements AttributeConverter<Ulid, String> {

    @Override
    public String convertToDatabaseColumn(Ulid ulid) {
        return ulid == null ? null : ulid.toString();
    }

    @Override
    public Ulid convertToEntityAttribute(String s) {
        return s == null ? null : Ulid.from(s);
    }
}
