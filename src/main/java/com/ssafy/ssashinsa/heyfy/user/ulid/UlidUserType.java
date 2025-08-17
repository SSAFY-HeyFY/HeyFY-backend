package com.ssafy.ssashinsa.heyfy.user.ulid;

import com.github.f4b6a3.ulid.Ulid;
import org.hibernate.engine.spi.SharedSessionContractImplementor;
import org.hibernate.usertype.UserType;

import java.io.Serializable;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Types;

public class UlidUserType implements UserType<Ulid> {

    @Override
    public int getSqlType() {
        return Types.VARBINARY;
    }

    @Override
    public Class<Ulid> returnedClass() {
        return Ulid.class;
    }

    @Override
    public boolean equals(Ulid x, Ulid y) {
        return x == y;
    }

    @Override
    public int hashCode(Ulid x) {
        return x.hashCode();
    }

    @Override
    public Ulid nullSafeGet(ResultSet rs, int position, SharedSessionContractImplementor session, Object owner) throws SQLException {
        byte[] bytes = rs.getBytes(position);
        return bytes != null ? Ulid.from(bytes) : null;
    }

    @Override
    public void nullSafeSet(PreparedStatement st, Ulid value, int index, SharedSessionContractImplementor session) throws SQLException {
        if (value == null) {
            st.setNull(index, Types.VARBINARY);
        } else {
            st.setBytes(index, value.toBytes());
        }
    }

    @Override
    public Ulid deepCopy(Ulid value) {
        return value;
    }

    @Override public boolean isMutable() {
        return false;
    }

    @Override public Ulid replace(Ulid original, Ulid target, Object owner) {
        return original;
    }

    @Override public Serializable disassemble(Ulid value) {
        return value;
    }

    @Override public Ulid assemble(Serializable cached, Object owner) {
        return (Ulid) cached;
    }
}